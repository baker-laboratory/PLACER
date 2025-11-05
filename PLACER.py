import sys, os
import json
import time
import glob
import itertools
import argparse
import tempfile
import numpy as np
import pandas as pd
import networkx as nx
import gzip
import gc
import copy
import torch
import random
from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)
DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, DIR)
import modules.model as model
import modules.geometry as geometry
import modules.dataloader_pdb as dataloader
import modules.pdbparser as pdbparser
import modules.losses as losses
import modules.cifutils as cifutils
import modules.obutils as obutils
import modules.utils as utils
import modules.protocol as protocol
from modules.placer_input import PLACERinput


class PLACER():
    def __init__(self, weights_path=None):
        """
        Loads the model from checkpoint, and sets up methods for data loading and loss calculation
        Use PLACER().run( ... ) to run predictions.
        """
        self.__device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('# device: ', self.__device)
        
        ########################################################
        # 0. load the network
        ########################################################

        self.__checkpoint = f'{DIR}/weights/PLACER_model_1.pt'
        if weights_path is not None:
            self.__checkpoint = weights_path

        NPROC = os.cpu_count()
        if "OMP_NUM_THREADS" in os.environ:
            NPROC = int(os.environ["OMP_NUM_THREADS"])
        if "SLURM_CPUS_ON_NODE" in os.environ:
            NPROC = int(os.environ["SLURM_CPUS_ON_NODE"])

        print(f"Using checkpoint: {self.__checkpoint}")
        chk = torch.load(self.__checkpoint, map_location=self.__device)
        self.__params = chk["params"]

        self.__DataLoader = dataloader.PDBDataset(csv = f'{DIR}/data/test.csv',  # mock dataset for inference
                                           ncpu = NPROC, world_size=1, rank=0,
                                           params=self.__params['DATALOADER']['featurizer'])

        self.__net = model.PLACER_network(**self.__params['NETWORK'],
                                          dims1d = self.__DataLoader.dataset.dataset.dims1d,
                                          dims2d = self.__DataLoader.dataset.dataset.dims2d).to(self.__device)

        state_dict = {k[7:] if k.startswith('module.') else k : v 
                      for k,v in chk["model"].items()}
        self.__net.load_state_dict(state_dict)
        self.__net.eval()
        nvar = sum(p.numel() for p in self.__net.parameters() if p.requires_grad)
        print('# variables: ', nvar)
        
        ## Loading CCD database
        self.__CCD_DB = None
        with gzip.open(f"{DIR}/data/ligands.json.gz", "rt", encoding="utf-8") as lgnds:
            self.__CCD_DB = json.load(lgnds)

        ## Storage of user inputs
        self.__fixed_ligand_noise = self.__params['DATALOADER']['featurizer']['sigma_bb']  # this can be changed through PLACERinput
        self.__verbose = True

        self.__Losses = losses.StructureLossesPDB(terms=["fape", "lddt", "rmsd", "kabsch"],
                                                  huber=self.params()['TRAINING']['HUBER'],
                                                  fapecut=self.params()['TRAINING']['FAPECUT'])

        pass


    ### GETTERS ###
    def net(self):
        """
        Returns the loaded model
        """
        return self.__net

    def device(self):
        """
        Returns the device CPU/GPU that the model is loaded to
        """
        return self.__device

    def params(self):
        """
        Returns the configuration parameters loaded from the checkpoint
        """
        return self.__params

    def dataloader(self):
        """
        Returns the instanciated dataloader which is used to parse/create PLACER network inputs
        """
        return self.__DataLoader
    
    def losses(self):
        """
        Returns the instanciated losses calculator of the PLACER network.
        """
        return self.__Losses
    
    def mols(self):
        """
        Returns the internally stored CCD ligands dictionary.
        """
        return self.__CCD_DB

    ### SETTERS / GETTERS ###
    def verbose(self, verbose=None):
        if verbose is None:
            return self.__verbose
        assert isinstance(verbose, bool)
        self.__verbose = verbose



    ### This function runs PLACER model on the input, and produces an output dictionary ###
    def run(self, input_object, nsamples: int, save_recycling_data=False, save_iter_features=False):
        """
        This function executes PLACER predictions

        Arguments:
            input_object (PLACER.PLACERinput)
                an object that defines input data for PLACER

            nsamples (int)
                How many iterations PLACER will run
            
            save_recycling_data (bool)
                Stores all outputs from the recycling process into the output dictionary.
                This includes predicted coordinates from each recycling step, and predicted confidences (pde, plddt, dev).
                The values are stored in outputs[N] under keys "Xs", "Ds", "plDDTs", "pDEVs"
                Beware, this will require plenty (>32gb) of system memory to store!
                Also >10GB GPU memory is likely required.

            save_iter_features (bool)
                Stores PLACER model inputs at each iteration into the output dictionary under outputs[N]["item"].
                It includes the atom graph (G), corrupted (X) and true (Y) coordinates, 1D and 2D features (f1d, f2d).
                This will start using significant amounts of system memory when producing >200 models.

        Returns:
            outputs (dict)
                Output dictionary contains the results of each prediction iteration as dictionaries.
                Each iteration dictionary contains the keys:
                dict_keys(['label', 'model_idx', 'fape', 'lddt', 'rmsd', 'kabsch', 'prmsd', 'plddt', 'plddt_pde', 'model', 'item'])

                outputs[N]["model"] contains a string representation of the generated model

        """

        label = input_object.name()
        if label is None:
            label = "structure"

        # Adding user-defined entities to the residue library
        if len(input_object.get_custom_residues()) != 0:
            for k in input_object.get_custom_residues():
                assert k not in self.mols().keys(), f"Residue {k} already in database, please choose a different name3."
            self.__CCD_DB.update(input_object.get_custom_residues())

        # Should these by conditionally instanciated?
        CIFParser = cifutils.CIFParser(skip_res=["HOH"]+input_object.skip_ligands(), mols=self.mols())
        PDBParser = pdbparser.PDBParser(skip_res=["HOH"]+input_object.skip_ligands(), mols=self.mols())


        """
        0. INPUT PREPROCESSING
        """
        ligand_reference = None
        if input_object.ligand_reference() is not None:
            ligand_reference = protocol.parse_ligand_reference(input_object.ligand_reference(), self.mols())

        ## Parsing user input structure
        chains, obmol = protocol.parse_input_structure(input_object, ligand_reference=ligand_reference,
                                                       pdbparser=PDBParser, cifparser=CIFParser)


        ### Parsing user choices about fixed ligands and to-be-prediced ligands ###
        fixed_ligand_noise = self.__fixed_ligand_noise
        if input_object.fixed_ligand_noise() is not None:
            fixed_ligand_noise = input_object.fixed_ligand_noise()

        # the fixed_ligands list is ligands that are not part of the user-requested predicted_ligands.
        # These fixed_ligands may or may not be part of the actualy prediction crop, depending on how close they are to the predict_ligands
        ligands_in_chains, fixed_ligands = protocol.parse_fixed_ligand_input(input_object, chains)

        if self.verbose():
            print(f"Keeping these ligands fixed during prediction IF they are in the crop: {fixed_ligands}.")

        if input_object.predict_multi() is True:
            assert len(ligands_in_chains) > 1, "predict_multi not usable if N_ligands <= 1"

        # Making sure that crop and corruption center inputs are all string, if input structure is CIF
        # That's because cifutils parses the atom names as all-string, but pdbparser sets residue number as int
        if input_object.cif() is not None and (input_object.corruption_centers() is not None or input_object.crop_centers() is not None):
            if input_object.corruption_centers() is not None:
               input_object.corruption_centers([cntr if len(cntr)==3 else (cntr[0],str(cntr[1]),cntr[2],cntr[3]) for cntr in input_object.corruption_centers()])
            if input_object.crop_centers() is not None:
               input_object.crop_centers([cntr if len(cntr)==3 else (cntr[0],str(cntr[1]),cntr[2],cntr[3]) for cntr in input_object.crop_centers()])



        ### Defining some parameters for PLACER inference ###
        recycles = [1]*self.params()['TRAINING']['NRECYCLES']
        input_keys = ['X', 'f1d', 'f2d', 'separation', 'bonds', 'bondlen', 'chirals', 'planars']
        terms = self.losses().terms+['prmsd','plddt','plddt_pde']
        terms = [x for x in terms if x not in ["ldist"]]


        counter = 0
        if self.verbose():
            chains_str = ''.join(['%s:%s/%d/%d; '%(k,v.type,len(v.atoms),len(v.bonds)) for k,v in chains.items()])
            print('--> %d %s {ID:type/atoms/bonds} : {%s}'%(counter,label,chains_str[:-2]))

        outputs = {}

        """
        1. RUNNING PREDICTIONS
        """
        for sample in range(nsamples):

            output = {key: None for key in ['label','model_idx']+terms}
            output.update({t: None for t in self.losses().terms})

            start_time = time.time()

            # crop around a small molecule
            cropped_atoms,center = protocol.build_crop(self.dataloader(), input_object, chains, obmol, fixed_ligands)

            # Doing some quality checking when user has defined additional bonds between atoms
            if input_object.bonds() is not None:
                # hacky edit, if input is CIF, then the residue number needs to be string
                if input_object.cif() is not None:
                    input_object.bonds([[(bnd[0][0], str(bnd[0][1]), bnd[0][2], bnd[0][3]), (bnd[1][0], str(bnd[1][1]), bnd[1][2], bnd[1][3]) , bnd[2]] for bnd in input_object.bonds()])
                # Making sure that bonded atoms are actually in the crop. Selecting centers until that is the case
                while not all([ all([atm in [_x.name for _x in cropped_atoms] for atm in bond_atms[:2]]) for bond_atms in input_object.bonds() ]):
                    print("Bonded atoms not in crop, recalculating the crop.")
                    cropped_atoms,center = protocol.build_crop(self.dataloader(), input_object, chains, obmol, fixed_ligands)


            # This would be a good one to print with verbose output, but it still kind of clutters stdout
            # if self.verbose():
            #     print(f"Using centers: {[ctr.name if hasattr(ctr, 'name') else ctr for ctr in center]}")

            # exclude small molecule atoms from the crop, if requested
            if input_object.exclude_sm() == True:
                sm_chids = [ch for ch in chains if chains[ch].type == "nonpoly"]
                to_exclude = []
                for ch in sm_chids:
                    to_exclude += set(chains[ch].atoms.keys())
                cropped_atoms = [a for a in cropped_atoms if a.name not in to_exclude]

            # convert the crop to nx.Graph
            G = self.dataloader().dataset.dataset.get_atom_graph(chains, [], cropped_atoms)
            L = len(G)


            # corrupt bonded neighbourhood around the center
            cutoff = 20  # params['DATALOADER']['featurizer']['maskrad']
            if input_object.exclude_sm() == False and input_object.target_res() is None:
                # Implemented below for multicenter prediction
                H = G.subgraph(nx.single_source_shortest_path_length(G, center[0].name, cutoff=cutoff))
                for ctr in center[1:]:
                    try:
                        H = nx.compose(*[H, G.subgraph(nx.single_source_shortest_path_length(G, ctr.name, cutoff=cutoff))])
                    except nx.exception.NodeNotFound as e:
                        print("ERROR!!", e)
                        print(f"Atom {ctr.name} is not in the crop!")
                        if input_object.crop_centers() is not None:
                            print("User defined crop centers were used - there's a chance that the randomly picked ligand is not in the user-defined crop.")
                            print("Please use `predict_ligand` to define which ligands should be considered for prediction.")
                        sys.exit(1)
                nx.set_node_attributes(H,False,'is_bb')

                corrupted = {n[1]['index'] for n in G.nodes(data=True) if (n[0] in H.nodes) and (n[1]['atom'].occ>0.0)}
                corrupted.update({n[1]['index'] for n in G.nodes(data=True) if n[1]['corrupted']==True})

                corrupted = list(corrupted)
                corrupted.sort()
            else:
                corrupted = []
                H = None
            item = {'corrupted' : torch.tensor(corrupted)}

            # add bond(s)
            if input_object.bonds() is not None:
                for (aname, bname, bondlen) in input_object.bonds():  # [(ch, resno, name3, atomname), (ch, resno, name3, atomname)]
                    assert aname in G.nodes
                    assert bname in G.nodes
                    try:
                        a = G.nodes[aname]['atom']
                        b = G.nodes[bname]['atom']
                        G.nodes[aname]['atom'] = a._replace(nhyd=max(a.nhyd-1,0))
                        G.nodes[bname]['atom'] = b._replace(nhyd=max(b.nhyd-1,0))
                        G.add_edge(aname,bname,bond=cifutils.Bond(
                            a=aname,
                            b=bname,
                            aromatic=False,
                            in_ring=False,
                            order=1,  # should this be user-definable?
                            intra=False,
                            length=bondlen) )

                    except:
                        sys.exit(f'ERROR: Cannot add bond {(aname, bname)}')


            # map corrupted atoms to closest anchors
            pairs,to_perturb = self.dataloader().dataset.dataset.map_to_anchors(G)

            ### randomize starting coords:
            #  - perturb extra anchors
            for node in to_perturb:
                Y = G.nodes[node]['Y']
                G.nodes[node]['X0'] = Y + torch.randn_like(Y)*self.params()['DATALOADER']['featurizer']['sigma']

            #  - perturb backbone
            for node in set(p[1] for p in pairs):
                Y = G.nodes[node]['Y']
                G.nodes[node]['X0'] = Y + torch.randn_like(Y)*self.params()['DATALOADER']['featurizer']['sigma_bb']

            #  - initialize the rest based on closest anchors
            for node,anchor in pairs:
                # If user has defined corruption centers then set the anchor coordinate to 
                # be the one that was selected before for the crop center associated with this ligand
                if H is not None and anchor in H and input_object.corruption_centers() is not None:
                    for ctr in center:
                        if ctr.name[:3] == anchor[:3]:
                            G.nodes[anchor]['X0'] = torch.tensor(ctr.xyz)
                X0 = G.nodes[anchor]['X0']

                if node==anchor:
                    # If the "ligand" is just a single atom then that anchor will be corrupted to some degree with small noise.
                    if H is not None and len(H.nodes) == 1 and node in H.nodes:
                        G.nodes[node]['X'] = X0 + torch.randn_like(X0)*self.params()['DATALOADER']['featurizer']['sigma']
                    else:
                        G.nodes[node]['X'] = X0
                elif (node[0], node[2], int(node[1])) in fixed_ligands:
                    G.nodes[node]['X'] = G.nodes[node]["Y"] + torch.randn_like(X0)*fixed_ligand_noise
                else:
                    G.nodes[node]['X'] = X0 + torch.randn_like(X0)*self.params()['DATALOADER']['featurizer']['sigma']


            # get topology
            item.update(dataloader.Dataset.get_topology(chains,G))

            # get input features
            f1d,f2d = self.dataloader().dataset.dataset.get_features_new(G)
            separation = f2d[...,-1]
            qmask = torch.bernoulli(torch.full((L,),self.params()['DATALOADER']['featurizer']['maskrate_q'])).bool()
            hmask = torch.bernoulli(torch.full((L,),self.params()['DATALOADER']['featurizer']['maskrate_h'])).bool()
            f1d = self.dataloader().dataset.dataset.OneHotF1D_new(f1d)
            f2d = self.dataloader().dataset.dataset.OneHotF2D(f2d)

            # add corruption mask to f1d
            crpt = torch.zeros((L,1))
            if input_object.exclude_sm() == False and input_object.target_res() is None:
                crpt[item['corrupted']] = 1
            f1d = torch.cat([f1d,crpt],dim=-1)


            item.update({
                'f1d' : f1d,
                'f2d' : f2d,
                'separation' : separation
            })

            # add coordinates and mask
            item.update({
                'X' : torch.stack([n[1]['X'] for n in G.nodes(data=True)]),
                'Y' : torch.stack([n[1]['Y'] for n in G.nodes(data=True)]),
                'observed' : torch.tensor([n[1]['atom'].occ>0 for n in G.nodes(data=True)])
            })

            # put tensors on device
            for k,v in item.items():
                if k in {'label','G'}:
                    pass
                elif isinstance(v, torch.Tensor):
                    item[k] = v.to(self.device())
                elif isinstance(v, list):
                    item[k] = [vi.to(self.device()) for vi in v]

            # run PLACER
            with torch.no_grad():
                Xs,Ds,plDDTs,pDEVs = self.net()(**{k:item[k] for k in input_keys}, recycles=recycles, save_recycling_data=save_recycling_data)


            # calculate losses
            with torch.no_grad():
                loss = self.losses().get_losses(Xs=Xs[-1:],
                                                 Ds=Ds[-1:],
                                                 Y=item['Y'],
                                                 topology=item,
                                                 plDDTs=plDDTs[-1:],
                                                 pDEVs=pDEVs[-1:])
            loss = loss.detach().tolist()

            with torch.no_grad():
                if input_object.exclude_sm() == False and input_object.target_res() is None:
                    sel = item['corrupted']
                else:
                    sel = torch.arange(L,device=self.device())
                plddt_pde = utils.get_plddt_pde(Xs[-1],Ds[-1],sel).item()
                plddt = utils.get_plddt(plDDTs[-1],sel).item()
                prmsd = utils.get_prmsd(pDEVs[-1],sel).item()

            # Removing the iteration outputs from GPU memory and storing them into output dict
            if save_recycling_data:
                output["Xs"] = [X.detach().cpu().numpy() for X in Xs]
                output["Ds"] = [X.detach().cpu().numpy() for X in Ds]
                output["plDDTs"] = [X.detach().cpu().numpy() for X in plDDTs]
                output["pDEVs"] = [X.detach().cpu().numpy() for X in pDEVs]

            if self.verbose():
                loss_str = self.losses().get_print_str([loss])
                loss_str += " %s= %.5f "%('prmsd',prmsd)
                loss_str += " %s= %.5f "%('plddt',plddt)
                loss_str += " %s= %.5f "%('plddt_pde',plddt_pde)
                loss_str += " | %s= %.2fs "%('time',(time.time()-start_time))
                loss_str += " %s= %.2fgb "%('mem',(torch.cuda.max_memory_allocated()/1024.**3))

                print('model %4d : %s'%(sample+1,loss_str))
                sys.stdout.flush()

            loss = loss+[prmsd,plddt,plddt_pde]
            output['label'] = label
            output['model_idx'] = sample+1
            for key,value in zip(terms,loss):
                output[key] = value

            output["center"] = center  # returning corruption centers

            ## Creating a PDBstring of the generated model
            X = Xs[-1].detach().cpu()
            bfac = pDEVs[-1].detach().cpu()
            output["model"] = utils.create_pdbmodel(G, X, bfac, sample+1)

            ## Storing the data that was used to run the model in this iteration
            output["item"] = {}
            if save_iter_features is True:
                for k,v in item.items():
                    if k in {'label','G'}:
                        continue
                    elif isinstance(v, torch.Tensor):
                        output["item"][k] = v.detach().cpu()
                    elif isinstance(v, list):
                        output["item"][k] = [vi.detach().cpu() for vi in v]
            else:
                # Saving minimal data: predicted coordinates X, and list of corrupted atoms
                output["item"]["X"] = item["X"].detach().cpu()
                output["item"]["corrupted"] = item["corrupted"].detach().cpu()
                output["item"]["permuts"] = [vi.detach().cpu() for vi in item["permuts"]]
                output["item"]["frames"] = item["frames"].detach().cpu()


            outputs[sample] = output

        # Do these actually help?
        gc.collect()
        torch.cuda.empty_cache()
        return outputs


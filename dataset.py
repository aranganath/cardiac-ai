import os
import torch
from torch.utils.data import Dataset
from typing import Tuple
from pdb import set_trace
class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    def __init__(self, 
        VmData: torch.tensor,
        ECGData: torch.tensor,
        actTimeData: torch.tensor,
        datInd: list,
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence 
                        before any slicing. If univariate, data.size() will be 
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence. 
                     The sub-sequence is split into src, trg and trg_y later.  

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """
        
        super().__init__()

        self.datInd = datInd

        self.VmData = VmData

        self.ECGData = ECGData

        self.actTime = actTimeData

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len



    def __len__(self):
        return self.ECGData.shape[0]

    def __getitem__(self, idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first and last element of the i'th tuple in the list self.VmInds

        src_list, trg_list, trg_y_list = [], [], []
        inp_seq = self.ECGData[idx,:,:]
        tar_seq = self.VmData[idx,:,:]
        act_time = self.actTime[idx, :]
        src, trg, trg_y = self.get_src_trg(
            inp_sequence= inp_seq,
            target_sequence = tar_seq,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )
            

        
        return src, trg, trg_y, act_time
        
    def get_src_trg(
        self,
        inp_sequence: torch.Tensor,
        target_sequence: torch.Tensor,
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence. 

        Args:

            sequence: tensor, a 1D tensor of length n where 
                    n = encoder input length + target sequence length  

            enc_seq_len: int, the desired length of the input to the transformer encoder

            target_seq_len: int, the desired length of the target sequence (the 
                            one against which the model output is compared)

        Return: 

            src: tensor, 1D, used as input to the transformer model

            trg: tensor, 1D, used as input to the transformer model

            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss. 
        
        """

        # encoder input
        src = inp_sequence[:enc_seq_len,:]
        trg, trg_y = [], []

        for i in range(len(self.datInd)-1):
            start_trg_idx, end_trg_idx = self.datInd[i][0], self.datInd[i][1]
            start_trgy_idx, end_trgy_idx = self.datInd[i+1][0], self.datInd[i+1][1]
            trg.append(target_sequence[start_trg_idx:end_trg_idx, :])
            trg_y.append(target_sequence[start_trgy_idx:end_trgy_idx, :])
        trg = torch.stack(trg, axis = 0)
        trg_y = torch.stack(trg_y, axis = 0)
        return src, trg, trg_y


# 1D CNN dataset
class CNN1dDataset(Dataset):
    """
    Dataset class used for 1D CNN models.
    
    """
    def __init__(self, 
        VmData: torch.tensor,
        ECGData: torch.tensor,
        num_timesteps: int,
        scaling_ecg= 'none', 
        scaling_vm= 'none',
        # actTimeData: torch.tensor,
        # datInd: list,
        # enc_seq_len: int, 
        # dec_seq_len: int, 
        # target_seq_len: int
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence 
                        before any slicing. If univariate, data.size() will be 
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

        """
        
        super().__init__()
        for idx in range(ECGData.shape[0]):        
        # --------------- Normalization of ECG data -------------------
            if scaling_ecg.lower() in ('normalized', 'normalization'):           
                min_ECG = torch.Tensor([-5.9542925, -5.5712219, -3.57188496, -4.288217553657807, -4.222570670021488, 
                                        -3.156398588951382, -2.447856981540037, -2.3431330150222047, -2.514583015022205, 
                                        -2.310517893645093, -1.9708348277623609, -2.32984159187901]);
                max_ECG = torch.Tensor([4.5093912, 4.2175924, 4.23887649, 5.473125233494979, 3.555297089959163, 3.018831728255531, 
                                        2.984374763664689, 2.359350064582923, 2.205404624762396, 2.014215222417507, 
                                        1.7023503410995358, 2.128710478327503]); 
                ECGData[idx] = (ECGData[idx] - min_ECG)/(max_ECG - min_ECG)
            
            if scaling_ecg.lower() in ('normalized_unit', 'normalization_unit'):
                diff_ECG = torch.max(ECGData[idx],0)[0] - torch.min(ECGData[idx],0)[0]
                ECGData[idx] = ECGData[idx]/diff_ECG
        
            if scaling_ecg.lower() in ('normalized_zero', 'normalization_zero'):           
                min_ECG = torch.Tensor([-5.9542925, -5.5712219, -3.57188496, -4.288217553657807, -4.222570670021488, 
                                        -3.156398588951382, -2.447856981540037, -2.3431330150222047, -2.514583015022205, 
                                        -2.310517893645093, -1.9708348277623609, -2.32984159187901]);
                max_ECG = torch.Tensor([4.5093912, 4.2175924, 4.23887649, 5.473125233494979, 3.555297089959163, 3.018831728255531, 
                                        2.984374763664689, 2.359350064582923, 2.205404624762396, 2.014215222417507, 1.7023503410995358, 
                                        2.128710478327503]); 
                ECGData[idx] = ECGData[idx]/(max_ECG - min_ECG)
        
            if scaling_ecg.lower() in ('standardized', 'standardization'): 
                mean_ECG = torch.Tensor([0.05337660278745852, 0.07800385312690625, 0.024627250339440174, -0.06569022815751291, 
                                         0.014374676023680694, 0.05131555153284552, -0.010237228016673849, -0.003921020891815309, 
                                         0.0022406280086986834, 0.009890627175816347, 0.028677603573294373, 0.03275114376978516]);
                std_ECG = torch.Tensor([0.5222836421647369, 0.46912923915554305, 0.37388222388756165, 0.4598741596222561, 
                                        0.3889260864136291, 0.33427537906748767, 0.3705529026825294, 0.2952769469657289, 
                                        0.25925690993927925, 0.22944537376676, 0.1901723804732019, 0.21860623885510164]);             
                ECGData[idx] = (ECGData[idx] - mean_ECG)/std_ECG 
        
            # --------------- Normalization of Vm data -------------------
            if scaling_vm.lower() in ('normalized', 'normalization'):           
                #min_Vm = torch.Tensor([-91.547070965423, -90.845797294849, -92.346791462634, -92.121723283769, -91.874668667206, 
                # -91.569205803862, -92.573625470269, -92.097113730859, -92.376835295723, -92.153128683318, -92.273465804737, 
                # -92.515685511545, -91.543891527166, -91.396827592875, -91.828461777899, -92.471531295168, -91.786368851851, 
                # -91.465714821866, -92.022001689223, -91.967088751123, -92.143020796117, -91.376752270548, -91.18683934801, 
                # -91.597161602795, -91.404739481162, -91.189016018189, -91.367746620938, -91.523352844722, -91.28281668588, 
                # -91.254153963368, -90.876479993773, -91.03747607014, -90.572647978259, -91.344873762032, -90.665037255865, 
                # -90.753762415098, -90.333292034956, -90.548678702044, -91.758256112755, -91.462933175156, -91.125926520588, 
                # -90.098179775406, -89.982124612163, -91.870810868507, -91.812413078483, -91.228393395603, -90.903016617148, 
                # -90.223510567776, -90.6423528435, -91.202203410457, -91.343243777991, -90.175414069802, -91.581373995542, 
                # -91.179774480065, -91.14930469691, -91.276732728796, -91.542486604186, -91.371611542998, -91.439310192837, 
                # -91.458498024566, -91.039596307374, -91.375934955627, -93.086583844575, -91.943027066598, -92.312558736778, 
                # -92.298525819992, -92.163025092035, -91.925944305106, -91.670715595924, -92.092823395069, -91.636319866411, 
                # -91.351636589078, -91.513032593541, -92.759957565172, -92.909840231414]);
                #max_Vm = torch.Tensor([46.248612640394, 49.487851490803, 49.446893077964, 49.436440277584, 49.407731608337, 
                # 49.419148769913, 49.351070038178, 49.530741145564, 49.468609033673, 49.418661992349, 49.412661638605, 49.439210059139, 
                # 49.4748682181, 49.356023962068, 49.414200321741, 49.408214439059, 49.477346732457, 49.40465524901, 49.339450491527, 
                # 49.427935235289, 49.403089385726, 49.457490510949, 49.152477436907, 49.126083653248, 49.494071838239, 49.18925460132, 
                # 49.435260208725, 49.44632171256, 49.139387403648, 49.406245558922, 49.426448104522, 49.104352111511, 48.987290952476, 
                # 48.976584491626, 49.124628836841, 49.036325003464, 49.297364351335, 49.23575427809, 49.478087212507, 49.366296897108, 
                # 49.252277385538, 49.125440811127, 49.200253621663, 49.224181420603, 49.241860273767, 49.180818724109, 49.329411225067, 
                # 49.283662392119, 49.336344104329, 49.204564827616, 48.985532589323, 49.451937575983, 49.196266981012, 49.146763295051, 
                # 49.221315098211, 49.173917830409, 49.184332499024, 49.170704682484, 49.087771790222, 49.084745546986, 49.152269095092, 
                # 49.365064723731, 49.249407543761, 49.373140706951, 49.462549062546, 49.127556631397, 49.182025848462, 49.232466484771, 
                # 49.040873382318, 49.370722057302, 49.46667861127, 49.435221512643, 48.016476126128, 49.21249261001, 48.079434835315]);            
                    min_Vm = -85.50618677
                    max_Vm = 50;          
                    VmData[idx] = (VmData[idx] - min_Vm)/(max_Vm - min_Vm)
            if scaling_vm.lower() in ('standardized', 'standardization'): 
                mean_Vm = torch.Tensor([-27.02603307424577, -26.42599561013216, -27.08525685162167, -27.072770564186968, 
                                        -27.243808102053098, -27.223278514349435, -27.083835486582533, -27.4110323325722, 
                                        -27.331364526546558, -27.292912836782325, -27.2623019983753, -27.309269034155918, 
                                        -27.338672021582383, -26.96347544613768, -27.158927183083865, -27.148198380852556, 
                                        -26.883271193655993, -27.257562163163644, -27.415302995291814, -26.94089741455307, 
                                        -26.380328171515497, -25.54398827003744, -23.23964030494721, -23.786303859496492, 
                                        -24.34536973736025, -22.657021527711205, -22.97386209700062, -24.401331456790917, 
                                        -22.85421172290724, -23.53197373751619, -22.835052832991884, -22.262005001437203, 
                                        -22.455317361231046, -22.89109758332462, -22.188754372239817, -23.255352662227548, 
                                        -24.477588837594464, -21.57554287398308, -23.21185108937846, -24.408397658499197, 
                                        -23.575402103548807, -22.70741982489576, -23.26547254656032, -22.556876909612093, 
                                        -23.473493713381124, -23.17731478310807, -23.3978924708568, -24.21465872516993, 
                                        -24.124127267631177, -23.007716729212664, -21.946535996882478, -23.82175334518444, 
                                        -22.123872753464493, -22.58709327049051, -22.922965821427884, -21.81210974698793, 
                                        -23.108769012452854, -23.667390240852303, -23.176059221741134, -24.46574628391638, 
                                        -24.280569203528557, -24.02632635847834, -24.47809251732797, -24.53259190412837, 
                                        -21.119403077336877, -23.46734272271838, -22.150611059141283, -20.056483455611414, 
                                        -23.7378480642948, -25.00461526170495, -25.54193419748148, -26.139503214725956, 
                                        -24.044751574496654, -18.106829931867534, -24.050961364158532]);
                std_Vm = torch.Tensor([47.641536695345756, 48.06899998911084, 48.07414930671482, 48.11237730036997, 
                                       48.129058287035924, 48.0640893651176, 48.04744258911431, 48.13517961142586, 48.11835490821148, 
                                       47.962570832683326, 48.045036073334025, 48.10542334800333, 48.1246977818833, 
                                       48.050984724069615, 48.150803515787764, 48.06273803810126, 48.14493033200949, 
                                       48.12373790487682, 47.94787316863915, 48.02717365919841, 48.16793355687098, 47.43404028501432, 
                                       47.69275264593347, 47.429461227850354, 47.27577038148137, 47.483485304378185, 
                                       47.35513468894153, 47.476477552939, 47.40493732227091, 47.36480405400306, 47.27410640503226, 
                                       47.326784740079724, 47.14305866446591, 47.14583848299832, 47.199156343425514, 
                                       47.34212272065462, 47.28098038286131, 47.327434856895884, 47.41420703719095, 
                                       47.40323474130151, 47.35621333118535, 47.23712532841057, 47.27712443301808, 47.32096036086348, 
                                       47.36162912212813, 47.254781765575274, 47.51120274336878, 47.573638059579956, 
                                       47.39408888626409, 47.22356400053576, 47.1674285402863, 47.07278873397787, 47.37766720677062, 
                                       47.180738258200996, 47.107764513714976, 47.39300851962479, 47.13585147999253, 
                                       47.44995913959363, 47.58919183197946, 47.60075423860692, 47.519772776916206, 
                                       47.668316011259535, 47.64926124185182, 47.36382151594125, 47.22019575795546, 
                                       47.17374251749065, 47.035487361419165, 47.01706929688778, 47.22759542323285, 
                                       47.63130076314641, 47.79796415558373, 47.84059288765446, 47.06397431137122, 
                                       46.780332763572964, 47.112581773104374]);
                VmData[idx] = (VmData[idx] - mean_Vm)/std_Vm         
        
        self.VmData = VmData

        self.ECGData = ECGData

        self.num_timesteps = num_timesteps

    def __len__(self):
        return self.ECGData.shape[0]

    def __getitem__(self, idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first and last element of the i'th tuple in the list self.VmInds

        src_list, trg_list, trg_y_list = [], [], []
        inp_seq = torch.zeros_like(self.ECGData[idx,:,:].T)
        inp_seq[:,:self.num_timesteps] = self.ECGData[idx,:self.num_timesteps,:].T
        tar_seq = self.VmData[idx,:,:].T
        # act_time = self.actTime[idx, :]
        # src, trg, trg_y = self.get_src_trg(
        #     inp_sequence= inp_seq,
        #     target_sequence = tar_seq,
        #     enc_seq_len=self.enc_seq_len,
        #     dec_seq_len=self.dec_seq_len,
        #     target_seq_len=self.target_seq_len
        #     )
            

        
        return inp_seq, tar_seq
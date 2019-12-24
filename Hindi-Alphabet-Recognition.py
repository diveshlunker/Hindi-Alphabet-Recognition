import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
import matplotlib.pyplot as plt

REBUILD_DATA = True

class HindiAlphabets():
    IMG_SIZE = 50
    KA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_1_ka"
    KHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_2_kha"
    GA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_3_ga"
    GHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_4_gha"
    KNA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_5_kna"
    CHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_6_cha"
    CHHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_7_chha"
    JA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_8_ja"
    JHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_9_jha"
    YNA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_10_yna"
    TAA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_11_taamatar"
    THAA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_12_thaa"
    DAA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_13_daa"
    DHAA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_14_dhaa"
    ADNA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_15_adna"
    TA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_16_tabala"
    THA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_17_tha"
    DA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_18_da"
    DHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_19_dha"
    NA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_20_na"
    PA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_21_pa"
    PHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_22_pha"
    BA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_23_ba"
    BHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_24_bha"
    MA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_25_ma"
    YAW = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_26_yaw"
    RA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_27_ra"
    LA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_28_la"
    WAW = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_29_waw"
    SAW = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_30_motosaw"
    SHA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_31_petchiryakha"
    SAW = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_32_patalosaw"
    HA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_33_ha"
    CHHYA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_34_chhya"
    TRA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_35_tra"
    GYA = "C:/Users/Divesh/pytorch proj/Hindi-Alphabet-Recognition/DevanagariHandwrittenCharacterDataset/character_36_gya"
    
    IMG_SIZE = 50
    
    LABELS = {KA:1,KHA:2,GA:3,GHA:4,KNA:5,CHA:6,CHHA:7,JA:8,JHA:9,YNA:10,TAA:11,THAA:12,DAA:13,DHAA:14,ADNA:15,TA:16,THA:17,DA:18,DHA:19,NA:20,PA:21,PHA:22,BA:23,BHA:24,MA:25,YAW:26,RA:27,LA:28,WAW:29,SAW:30,SHA:31,SAW:32,HA:33,CHHYA:34,TRA:35,GYA:36}
    
    training_data = []
    
    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm( os.listdir(label)) :
                try:
                    path = os.path.join(label,f)
                    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))

                    self.training_data.append([np.array(img),np.eye(36)[self.LABELS[label]]])
                except Exception as e:
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)


    
if REBUILD_DATA:
    alphabet = HindiAlphabets()
    alphabet.make_training_data()
    
training_data = np.load("training_data.npy",allow_pickle=True)


plt.imshow(training_data[2][0],cmap = "gray")
plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,36)
    
    
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)
net = Net()


optimizer  = optim.Adam(net.parameters(),lr = 0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X  =X/255.0

y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)


train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))

BATCH_SIZE = 100
EPOCHS = 2

for epoch in range(EPOCHS):
    for i in tqdm(range(0,len(train_X),BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]
        
        net.zero_grad()
        outputs = net(batch_X)
        
        loss = loss_function(outputs,batch_y)
        loss.backward()
        optimizer.step()
print(loss)


correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct+=1
        total+=1
print(" Accuracy = ",round(correct/total,3))

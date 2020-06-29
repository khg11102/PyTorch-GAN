# PyTorch-GAN-MNIST
import torch
import torch.nn as nn #Neural Networks
import torch.optim as optim #최적화 알고리즘을 구현하는 패키지 
import torchvision.utils as utils
import torchvision.datasets as dsets #널리 사용되는 데이터세트, 모델, 아키텍처 
import torchvision.transforms as transforms #이미지 변환 패키지

import numpy as np #행렬, 배열 계산
from matplotlib import pyplot as plt #데이터 시각화 패키지 


is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(device)

#transforms.Compose 데이터 전처리하기 위한 패키지
standardizator = transforms.Compose([transforms.ToTensor(), #데이터 타입을 Tensor형태로 변형
                                    transforms.Normalize((0.5,), (0.5,)) #ToTensor()로 타입 변경시 0~1사의 값
                                    ]) 

# MNIST dataset
train_data = dsets.MNIST(root='data/', train=True, transform=standardizator, download=True) # root 경로지정, train 데이터를 받음, transforms 사전에 설정한 데이터 전처리형태, download 데이터 받기
test_data  = dsets.MNIST(root='data/', train=False, transform=standardizator, download=True)# root 경로지정, train 데이터를 받지않음, transforms 사전에 설정한 데이터 전처리형태, download 데이터 받기


batch_size = 200 #배치사이즈 
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True) #불러올 데이터, 배치사이즈, 셔플 
test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True) #불러올 데이터, 배치사이즈, 셔플

# print(train_data)
# print(test_data)

# dataiter = iter(train_data_loader)
# #첫번째 그룹 4개 이미지 데이터 획득
# train, label = dataiter.next()
# # 첫번째 4개 이미지의 데이터 Shape 확인
# print(train.shape)
# print('torch.size([200, 1, 28, 28]) = 200개의 배치, 1개의 채널, 28x28사이즈')

# #Input이미지에 대한 기하학적 변화
# from google.colab.patches import cv2_imshow
# !curl -o logo.png https://colab.research.google.com/img/colab_favicon_256px.png
# import cv2
# img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
# img1 = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)

# img1 = (img+1)/2

# cv2_imshow(img)
# cv2_imshow(img1)

# img1 = img1.squeeze() # 차원중 사이즈 1인 것을 찾아 제거한다.
# np_img1 = np.array(img1) # 이미지 픽셀을 넘파이 배열로
# plt.imshow(np_img1, cmap='gray') #그레이 스케일
# plt.axis('off')
# plt.show()

def imshow(img):
    img = (img+1)/2 #이미지 배열에다 형태학적 변화를 주기위해
    img = img.squeeze() # 차원중 사이즈 1인 것을 찾아 제거한다
    np_img = img.numpy() # 이미지 픽셀을 넘파이 배열로
    plt.imshow(np_img, cmap='gray') #그레이 스케일
    plt.show()

def imshow_grid(img): 
    img = utils.make_grid(img.cpu().detach()) #이미지 그리드를 만듭니다 #이미지 출력만을 위해서 cpu에 담고 추적을 방지
    img = (img+1)/2 #이미지 배열에다 형태학적 변화를 주기위해
    npimg = img.numpy() # 이미지 픽셀을 넘파이 배열로
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
example_mini_batch_img, example_mini_batch_label  = next(iter(train_data_loader))
imshow_grid(example_mini_batch_img[0:16]) # slice notation 0부터 16번까지의 Original MNIST img 출력

# print(next(iter(train_data_loader)))
# print(example_mini_batch_img)
# pix1 = example_mini_batch_img
# pix2 = example_mini_batch_img.numpy()
# print(pix2)
# print(pix2.shape)
# pix2 = example_mini_batch_img.squeeze()
# print(pix2)
# print(pix2.shape)

d_noise  = 100
d_hidden = 256

def sample_z(batch_size = 1, d_noise=100):
    return torch.randn(batch_size, d_noise, device=device)

print(torch.randn(batch_size, d_noise, device=device))

#Generator Net
G = nn.Sequential(
    nn.Linear(d_noise, d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden,d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, 28*28),
    nn.Tanh()
).to(device)

# 노이즈 생성하기
z = sample_z()
# 가짜 이미지 생성하기
img_fake = G(z).view(-1,28,28)
# 이미지 출력하기
imshow(img_fake.squeeze().cpu().detach()) #현재 계산에서 분리시키고 계산들은 추적되지 않음

# Batch SIze만큼 노이즈 생성하여 그리드로 출력하기
z = sample_z(batch_size)
img_fake = G(z) #enerator 모델을 통과한 200배치의 노이즈 벡터 이미지
imshow_grid(img_fake)
print(z.shape)
print(G(z).shape)
print(G(z).view(-1,28,28).shape) #-1은 자동으로 값을 가져온다 (200, 28, 28)
type(img_fake) # 텐서타입의 fake이미지가 생성된다

# # .view 이해하기
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)
# print('x.size =',x.size(),'y.size =' ,y.size(),'z.size =' ,z.size())

#Discriminator Net
D = nn.Sequential(
    nn.Linear(28*28, d_hidden), 
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, d_hidden),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, 1),
    nn.Sigmoid() #BCELoss를 사용하기 때문에 마지막엔 Sigmoid를 사용해줘야한다
).to(device)

print(D)
print(G(z).shape)
print(D(G(z)).shape)
print(D(G(z)[0:5]).transpose(0,1))

# x1 = torch.FloatTensor([0.31,-2,3])
# x2 = torch.log(x1)
# x3 = torch.log(1.-x1)
# loss_test1 = -1 * x2
# print(x2)
# print(x3)
# print(loss_test1)

#BinaryCrossEntorpyLoss
criterion = nn.BCELoss()

def run_epoch(generator, discriminator, _optimizer_g, _optimizer_d):
    
    generator.train()
    discriminator.train()

    for img_batch, label_batch in train_data_loader:
        
        img_batch, label_batch = img_batch.to(device), label_batch.to(device) 

        # ================================================  #
        # maximize V(D,G) = optimize discriminator (setting k to be 1)  #
        # ================================================  #

        # init optimizer
        _optimizer_d.zero_grad() # 가중치 0으로 초기화

        p_real = discriminator(img_batch.view(-1, 28*28))
        # print("p_real = ",p_real[0:16]) #p_real이미지의 기울기
        p_fake = discriminator(generator(sample_z(batch_size, d_noise)))
        # print("p_fake = ",p_fake[0:16]) #p_fake벡터의 기울기
        
        # ================================================  #
        #    Loss computation (soley based on the paper)    #
        # ================================================  #
        loss_real = -1 * torch.log(p_real)   # -1 for gradient ascending 
        loss_fake = -1 * torch.log(1.-p_fake) # -1 for gradient ascending 미분?
        loss_d    = (loss_real + loss_fake).mean() #전체 기울기의 평균
        # print(loss_real)
        # print(loss_fake)
        # print(loss_d)

        
        # ================================================  #
        #     Loss computation (based on Cross Entropy)     #
        # ================================================  #
        # loss_d = criterion(p_real, torch.ones_like(p_real).to(device)) + \    #
        #          criterion(p_fake, torch.zeros_like(p_real).to(device))       #
        
        # Update parameters
        loss_d.backward()
        _optimizer_d.step()

        # ================================================  #
        #        minimize V(discriminator,generator)        #
        # ================================================  #

        # init optimizer
        _optimizer_g.zero_grad()

        p_fake = discriminator(generator(sample_z(batch_size, d_noise)))
                
        # ================================================  #
        #    Loss computation (soley based on the paper)    #
        # ================================================  #
        
        # instead of: torch.log(1.-p_fake).mean() <- explained in Section 3
        loss_g = -1 * torch.log(p_fake).mean()

        # ================================================  #
        #     Loss computation (based on Cross Entropy)     #
        # ================================================  #
        # loss_g = criterion(p_fake, torch.ones_like(p_fake).to(device)) #

        loss_g.backward()
   
        # Update parameters
        _optimizer_g.step()

# #torch.sum 이해하기
# x = torch.tensor([[1, 2, 3],
#                  [4, 5, 6]])
# y = torch.tensor([[1, 2, 3],
#                  [4, 5, 6]])
# x.shape
# output = x + y
# print(output)
# print(torch.sum(output))
# print(torch.sum(output).shape)
# print(type(torch.sum(output)))

# #.item() 이해하기
# scl = torch.tensor(1)
# print(type(scl.item()))

# #복합 대입 연산자 += 이해하기
# #(a += b) = (a = a + b) 
# a = 1
# print(a)
# a += 1
# print(a)
# a += 1
# print(a)

def evaluate_model(generator, discriminator):
    
    p_real, p_fake = 0.,0.
    
    generator.eval() #.eval()문자열로 표현된 식을 인수로 받아 연산할 수 있다
    discriminator.eval()
        
    for img_batch, label_batch in test_data_loader:
        
        img_batch, label_batch = img_batch.to(device), label_batch.to(device) 
        
        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch.view(-1, 28*28))).item())/10000. #.item() 1개의 원소를 가진 텐서를 스칼라로 만들때 사용하는 함수
            p_fake += (torch.sum(discriminator(generator(sample_z(batch_size, d_noise)))).item())/10000.#.item() 1개의 원소를 가진 텐서를 스칼라로 만들때 사용하는 함수
            #p_fake = smape_z(200, 100).item()/10000
            
    return p_real, p_fake

print("Generator model")
print(G.parameters)
print("Discriminator model")
print(D.parameters)

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)

init_params(G)
init_params(D)

optimizer_g = optim.Adam(G.parameters(), lr = 0.0002) #G.model 파라미터
optimizer_d = optim.Adam(D.parameters(), lr = 0.0002) #D.model 파라미터

p_real_trace = []
p_fake_trace = []

EPOCHS = 100
for epoch in range(EPOCHS):
    
    run_epoch(G, D, optimizer_g, optimizer_d)
    p_real, p_fake = evaluate_model(G,D)
    
    p_real_trace.append(p_real)
    p_fake_trace.append(p_fake) 
    
    if((epoch+1)% 10 == 0):
        print('(epoch %i/{}) p_real: %f, p_g: %f'.format(EPOCHS) % (epoch+1, p_real, p_fake))
        imshow_grid(G(sample_z(16)).view(-1, 1, 28, 28))

plt.plot(p_fake_trace, label='D(x_generated)') # p_real 데이터 Loss 
plt.plot(p_real_trace, label='D(x_real)') # p_Genrator 데이터 Loss
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

vis_loader = torch.utils.data.DataLoader(test_data, 16, True)
img_vis, label_vis   = next(iter(vis_loader))
imshow_grid(img_vis)

imshow_grid(G(sample_z(16,100)).view(-1, 1, 28, 28))

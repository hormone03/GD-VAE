#from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.distributions.gamma import Gamma

#from LogisticRegression_VAE import LogisticRegression
from tqdm import tqdm

from utils import train_classifier, evaluate_classifier
from utils import return_model_accurary
from Visualisations import t_SNE, describe_statistic_per_label, show_confusion_matrix


parser = argparse.ArgumentParser(description='Dir-VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)') #epoch was 10
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--category', type=int, default=10, metavar='K',
                    help='the number of categories in the dataset')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)  

device = torch.device("cuda" if args.cuda else "cpu")


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

ngf = 64
ndf = 64
nc = 1

def prior(K, alpha):
    """
    Prior for the model.
    :K: number of categories
    :alpha: Hyper param of Dir
    :return: mean and variance tensors
    """
    
    # Approximate to normal distribution using Laplace approximation
    a = torch.Tensor(1, K).float().fill_(alpha)
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # Parameters of prior distribution after approximation

class GD(nn.Module):
    def __init__(self):
        super(GD, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, args.category)
        self.fc22 = nn.Linear(512, args.category)

        self.fc3 = nn.Linear(args.category, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.B = 1.0

        # Dir prior
        self.prior_alpha = torch.tensor(0.01).cuda()
        self.prior_beta = torch.tensor(0.02).cuda() #map(nn.Parameter, prior(args.category, 0.3)) # 0.3 is a hyper param of Dirichlet distribution
        self.prior_alpha.requires_grad = False
        self.prior_beta.requires_grad = False


    def encode(self, x):
        conv = self.encoder(x);
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, gauss_z):
        dir_z = F.softmax(gauss_z,dim=1) 
        # This variable (z) can be treated as a variable that follows a Dirichlet distribution (a variable that can be interpreted as a probability that the sum is 1)
        # Use the Softmax function to satisfy the simplex constraint
        h3 = self.relu(self.fc3(dir_z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)
    
    def gamma_h_boosted(self, epsilon, u, alpha,model_B):
        #(eps,u,alpha,batch_size)
        epsilon = epsilon.to(device)
        u = u.to(device)
        alpha = alpha.to(device)
        #Me: this calculate z inside section 3.4, z_tilder there now implies EQ2
        #Me: Note that all the lines till u_power generate u
        """
        Reparameterization for gamma rejection sampler with shape augmentation.
        """
        #B = u.shape.dims[0] #u has shape of alpha plus one dimension for B
        #Me: Note that alpha shape is not noumber of topic
        B = u.shape[0]
        K = alpha.shape[1]#(batch_size,K)
        r = torch.arange(0,B) #Me: range(0, n), n=B here 
        #Me: reshape r
        rm = (torch.reshape(r,[-1,1,1])).type(torch.FloatTensor).to(device) #dim Bx1x1
        #Me: tile expand and Copy a Tensor: tf.tile(input,multiples,name=None)
        #Me: xy = tf.tile(xs, multiples = [2, 3]) means repeat generate twice in x-ddirection, 3 times in y-direction
        #Me:https://www.tutorialexample.com/understand-tensorflow-tf-tile-expand-a-tensor-tensorflow-tutorial/
        alpha_vec = torch.reshape(torch.tile(alpha,(B,1)),(model_B,-1,K)) + rm #dim BxBSxK + dim Bx1
        alpha_vec = alpha_vec.to(device)
        u_pow = torch.pow(u,1./alpha_vec)+1e-10
        gammah = self.gamma_h(epsilon, alpha + torch.tensor(B))
        return torch.prod(u_pow,axis=0)*gammah

    
    def calc_epsilon(self,gamma,alpha):
        return torch.sqrt(9.*alpha-3.)*(torch.pow(gamma/(alpha-1./3.),1./3.)-1.)

    def my_random_gamma(self,shape, alpha, beta=1.0):

        alpha = torch.ones(shape).to(device) * alpha
        beta = torch.ones(shape).to(device) * torch.tensor(beta).to(device)
        
        gamma_distribution = Gamma(alpha, beta)
    
        return gamma_distribution.sample()
    
    def gamma_h(self, epsilon, alpha):
        #Me: gamma_h is from equ_2: z=h_gamma(eps, alpha)= EQ2
        """
        Reparameterization for gamma rejection sampler without shape augmentation.
        """
        b = alpha - 1./3.
        c = 1./torch.sqrt(9.*b)
        v = 1.+epsilon*c
        
        return b*(v**3) 


    def reparameterize(self, alpha, beta):
        std = torch.exp(0.5*beta)
        eps = torch.randn_like(std)
        return alpha + eps*std
    
    def _reparameterize(self, alpha, beta):
        alpha = torch.max(torch.tensor(0.0001), alpha)
        beta = torch.max(torch.tensor(0.0001), beta)
        
        gam1 = torch.squeeze(self.my_random_gamma(shape = (1,), alpha = alpha+ torch.tensor(self.B)))
        gam2 = torch.squeeze(self.my_random_gamma(shape = (1,), alpha = beta+ torch.tensor(self.B)))
        
        eps = (self.calc_epsilon(gam1,alpha+torch.tensor(self.B))).detach()
        eps2 = (self.calc_epsilon(gam2,beta+torch.tensor(self.B))).detach()
        #uniform variables for shape augmentation of gamma
        u = torch.rand(1,alpha.shape[0],10)
        u2 = torch.rand(1,beta.shape[0],10)
        #u=torch.FloatTensor(self.n_sample, self.B).uniform_(1, self.n_topic)
        #this is the sampled gamma for this document, boosted to reduce the variance of the gradient
        doc_vec = self.gamma_h_boosted(eps,u,alpha,u.shape[0])
        doc_vec2 = self.gamma_h_boosted(eps2,u2,beta,u2.shape[0])
        
        z = doc_vec/(doc_vec+doc_vec2)
        
        z = torch.div(doc_vec,torch.reshape(torch.sum(doc_vec,1), (-1, 1)))
        
        return z 
    
    def forward(self, x):
        alpha, beta = self.encode(x)
        z = self.reparameterize(alpha, beta) 
        # gause_z is a variable that follows a multivariate normal distribution
        # Inputting gause_z into softmax func yields a random variable that follows a Dirichlet distribution (Softmax func are used in decoder)
        dir_z = F.softmax(z,dim=1) # This variable follows a Dirichlet distribution
        return self.decode(z), alpha, beta, z, dir_z

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, encoderAlpha, encoderBeta, K):
        NLL = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        #NLL = -(x.view(-1, 784) * (recon_x.view(-1, 784) + 1e-10).log()).sum(1) 
        
        encoderAlpha = torch.max(torch.tensor(0.0001), encoderAlpha)
        encoderBeta = torch.max(torch.tensor(0.0001), encoderBeta)
        prior_alpha = self.prior_alpha.expand_as(encoderAlpha)
        prior_beta = self.prior_beta.expand_as(encoderBeta)
        
        alphaDiff = encoderAlpha - prior_alpha
        betaDiff = encoderBeta - prior_beta 
        priorsParamsSum = prior_alpha + prior_beta
        encodeParamsSum = encoderAlpha + encoderBeta
        
        numerator = torch.lgamma(priorsParamsSum) + torch.lgamma(prior_alpha) + torch.lgamma(prior_beta)
        denomerator = torch.lgamma(encodeParamsSum) + torch.lgamma(encoderAlpha) + torch.lgamma(encoderBeta)
        firstTerm = numerator - denomerator
        secondTerm = alphaDiff * (torch.digamma(encoderAlpha) - torch.digamma(priorsParamsSum))
        thirdTerm = betaDiff * (torch.digamma(encoderBeta) - torch.digamma(priorsParamsSum))
        KLD = torch.sum((firstTerm + secondTerm + thirdTerm), dim=1)
        #print('*' * 30)
        #print(KLD)
        return NLL + KLD

model = GD().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
Rg_model = LogisticRegression(random_state=16, solver='lbfgs', max_iter=3000)

def train(epoch, data, train_loss):
    data = data.to(device)
    optimizer.zero_grad()
    recon_batch, alpha, beta, gauss_z, dir_z = model(data)
    #logistic_z = gauss_z.cpu().detach()
    #Rg_model = Rg_model.fit(data.view(-1, 784).cpu(), Y.cpu())
    #RegAccuracy = Rg_model.score(recon_batch.view(-1, 784).detach().cpu(), Y.cpu())
    loss = model.loss_function(recon_batch, data, alpha, beta, args.category)
    loss = loss.mean().to(device)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        #print(f"gause_z:{gauss_z[0]}") # Variables following a normal distribution after Laplace approximation
        #print(f"dir_z:{dir_z[0]},SUM:{torch.sum(dir_z[0])}") # Variables that follow a Dirichlet distribution. This is obtained by entering gauss_z into the softmax function
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item() / len(data)))
        #print('Logistic model accuracy = {:.4f}'.format(RegAccuracy))

    #print('====> Epoch: {} Average loss: {:.4f}'.format(
         # epoch, train_loss / len(train_loader.dataset)))
    #print('====> Average logistic model accuracy = {}'.format(RegAccuracy))
    return recon_batch, train_loss

def trainLG(Rg_model, recon_batch, data):
    #logistic_z = gauss_z.cpu().detach()
    Rg_model = Rg_model.fit(data.view(-1, 784).cpu(), Y.cpu())
    RegAccuracy = Rg_model.score(recon_batch.view(-1, 784).detach().cpu(), Y.cpu())
    if batch_idx % args.log_interval == 0:
        print('Logistic model accuracy = {:.4f}'.format(RegAccuracy))
    return Rg_model

    

def test(epoch, data, test_loss):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        recon_batch_test, alpha, beta, gauss_z, dir_z = model(data)
        loss = model.loss_function(recon_batch_test, data, alpha, beta, args.category)
        test_loss += loss.mean()
        test_loss.item()
        if i == 0:
            n = min(data.size(0), 18)
            comparison = torch.cat([data[:n],
                                  recon_batch_test.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.cpu(),
                     'image/recon_' + str(epoch) + '.png', nrow=n)
    return recon_batch_test, test_loss

   
def testLG(Rg_model, recon_batch_test):
    model.eval()
    with torch.no_grad():
        RegAccuracy = Rg_model.score(recon_batch_test.view(-1, 784).detach().cpu(), label.cpu())
    return RegAccuracy
    

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train_loss = 0
        for batch_idx, (data, Y) in enumerate(train_loader):
            recon_batch, train_loss = train(epoch, data, train_loss)
            Rg_model = trainLG(Rg_model, recon_batch, data)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
        test_loss = 0
        for i, (data, label) in enumerate(test_loader):
            recon_batch_test, test_loss = test(epoch, data, test_loss)
            RegAccuracy = testLG(Rg_model, recon_batch_test)
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Test set for Logistic Regression accuracy: {:.4f}'.format(RegAccuracy))
        with torch.no_grad():
            sample = torch.randn(64, args.category).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),'image/sample_' + str(epoch) + '.png')
            
    show_confusion_matrix(test_loader, model, Rg_model, use_cuda=True)
     
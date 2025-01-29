'''
SOURCED FROM MY COLLABORATION IN TWARDLAB UCLA SUMMER 2025 - I KINDLY ASK THAT ANYONE SEEING THIS DEEPLEARNING2025 PROJECT NOT DISTRIBUTE OR SHARE THIS FILE PUBLICLY!

Rotation and reflection equivariance using moment kernels in 2D.

We support scalar fields, vector fields, convolution maps between them, batch norm, and nonlinearity.

TODO
----
Try to move if statements out of the forward method and somehow into the init method.

Consider adding in rotation but not reflection for 2D.

Build 3D.
'''
import torch
import numpy as np


class ScalarToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if out_channels == 0:
            self.forward = forward_empty
            return
        
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3.0*in_channels)) # TODO: use the right normalizatoin
        self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))       
    def forward_empty(self,x):
        ''' 
        Return an array that's the same size as the input but with 0 channels
        This can be used to concatenate with other arguments
        Note this requires a batch dimension
        TODO: compute the correct size with respect to padding and kernel size
        I'm not sure if this is a good approach.
        '''
        return torch.zeros((x.shape[0],0,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        
    def forward(self,x):
        # note this works with size 1 images, as long as padding is 0
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size        
        c = self.weights[...,self.inds]
        self.c = c
                        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias)
    
    
        
        
class ScalarToVector(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, out channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[:,None]
        Xhat = Xhat.repeat((out_channels,1,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        
        inds = inds - 1 # we will not use r=0.  the filter will get assigned a different number, but then multiplied by 0
        inds[inds==-1] = 0        
        self.register_buffer('inds',inds) # don't need a parameter for r=0, but this makes
        
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels)) # TODO: use the right normalizatoin
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):        
        # kernel size 1 needs to be a special case because self.inds is empty, the result is just 0
        # no padding allowed
        # note we assume square
        return torch.zeros(x.shape[0],self.out_channels*2,1,1,dtype=x.dtype,device=x.device)
    
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size          
        
        c = torch.repeat_interleave(self.weights,2,0)[...,self.inds]*self.Xhat
        self.c = c
        
        
        # for somme reason the output is not zero mean, has to do with padding
        # here's a better way
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)
        return torch.nn.functional.conv2d(tmp,c)
        
        

class ScalarToVector90(torch.nn.Module):
    '''This module adds an additional basis function with a 90 degree rotation'''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, out channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[:,None]
        X90hat = Xhat.flip(0)*torch.tensor([-1.0,1.0])[:,None,None,None]
        Xhat = Xhat.repeat((out_channels,1,1,1))
        X90hat = X90hat.repeat((out_channels,1,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)        
        self.register_buffer('X90hat',X90hat)
        
        inds = inds - 1 # we will not use r=0.  the filter will get assigned a different number, but then multiplied by 0
        inds[inds==-1] = 0        
        self.register_buffer('inds',inds) # don't need a parameter for r=0, but this makes
        
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels)) # 
        self.weights90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels)) # TODO: use the right normalizatoin
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):        
        # kernel size 1 needs to be a special case because self.inds is empty, the result is just 0
        # no padding allowed
        # note we assume square
        return torch.zeros(x.shape[0],self.out_channels*2,1,1,dtype=x.dtype,device=x.device)
    
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size          
        
        c = torch.repeat_interleave(self.weights,2,0)[...,self.inds]*self.Xhat
        c90 = torch.repeat_interleave(self.weights90,2,0)[...,self.inds]*self.X90hat
        self.c = c + c90
        
        
        # for somme reason the output is not zero mean, has to do with padding
        # here's a better way
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)
        return torch.nn.functional.conv2d(tmp,c)
        
        
                
        
            
        
        
        
def rotate_vector_and_image(x):
    with torch.no_grad():
        tmp = x.rot90(1,(-1,-2))
        tmp2 = tmp.clone()        
        for i in range(tmp.shape[1]//2):
            tmp2[:,i*2] = tmp[:,i*2+1]
            tmp2[:,i*2+1] = -tmp[:,i*2]

    return tmp2

def rotate_vector(x):
    with torch.no_grad():  
        tmp = x.clone()
        tmp2 = x.clone()        
        for i in range(tmp.shape[1]//2):
            tmp2[:,i*2] = tmp[:,i*2+1]
            tmp2[:,i*2+1] = -tmp[:,i*2]

    return tmp2




class VectorToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[None]
        Xhat = Xhat.repeat((1,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        inds = inds - 1
        inds[inds<0] = 0
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2)) # TODO: use the right normalizatoin
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))        
        else:
            self.bias = None
            
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):
        # size 1 is a special case because there are no parameters, just return 0 + bias
        # self.ind is empty
        return torch.zeros(x.shape[0],self.out_channels,1,1,dtype=x.dtype,device=x.device) + self.bias[...,None,None]
    
    def forwardg1(self,x):
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        c = torch.repeat_interleave(self.weights[...,self.inds],2,1)*self.Xhat                
        self.c = c
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias) 
        

class VectorToScalar90(torch.nn.Module):
    ''' In this version we include the extra basis rotated by 90 degrees'''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[None]
        X90hat = Xhat.flip(1)*torch.tensor([-1.0,1.0])[None,:,None,None]
        Xhat = Xhat.repeat((1,in_channels,1,1))
        X90hat = X90hat.repeat((1,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        inds = inds - 1
        inds[inds<0] = 0
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('X90hat',X90hat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2)) # TODO: use the right normalizatoin
        self.weights90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2)) # TODO: use the right normalizatoin
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))        
        else:
            self.bias = None
            
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    
    def forwarde1(self,x):
        # size 1 is a special case because there are no parameters, just return 0 + bias
        # self.ind is empty
        return torch.zeros(x.shape[0],self.out_channels,1,1,dtype=x.dtype,device=x.device) + self.bias[...,None,None]
    
    def forwardg1(self,x):
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        c = torch.repeat_interleave(self.weights[...,self.inds],2,1)*self.Xhat + torch.repeat_interleave(self.weights90[...,self.inds],2,1)*self.X90hat
        self.c = c
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias) 
                
        
        
class VectorToVector(torch.nn.Module):
    '''Question, should I separate these into two types and interleave them somehow rather than combining them'''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)
        XhatXhat = Xhat[None,:]*Xhat[:,None]        
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
        # special case if kernel is size 1
        # print(x.shape)
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    def forwarde1(self,x):
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity[...,self.indsidentity],2,0),2,1)*self.identity
        self.cidentity = cidentity
        return torch.nn.functional.conv2d(x,cidentity)
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        cxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxx,2,0),2,1)[...,self.indsxx]*self.XhatXhat
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity,2,0),2,1)[...,self.indsidentity]*self.identity
        c = cxx + cidentity
        self.c = c
        self.cxx = cxx
        self.cidentity = cidentity
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c) # no bias when output is vector

    
class VectorToVector90(torch.nn.Module):
    '''Question, should I separate these into two types and interleave them somehow rather than combining them
    This one uses the extra basis functions rotated by 90 degrees.
    '''
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)        
        X90hat = Xhat.flip(0)*torch.tensor([-1.0,1.0])[...,None,None]
        # now there are 4
        XhatXhat = Xhat[None,:]*Xhat[:,None]
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        X90hatXhat = X90hat[None,:]*Xhat[:,None]
        X90hatXhat = X90hatXhat.repeat((out_channels,in_channels,1,1))
        XhatX90hat = Xhat[None,:]*X90hat[:,None]
        XhatX90hat = XhatX90hat.repeat((out_channels,in_channels,1,1))
        X90hatX90hat = X90hat[None,:]*X90hat[:,None]
        X90hatX90hat = X90hatX90hat.repeat((out_channels,in_channels,1,1))
        
        
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('X90hatXhat',X90hatXhat)
        self.register_buffer('XhatX90hat',XhatX90hat)
        self.register_buffer('X90hatX90hat',X90hatX90hat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsx90x = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxx90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsx90x90 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
        # special case if kernel is size 1
        # print(x.shape)
        if x.shape[-1] == 1:
            self.forward = self.forwarde1
        else:
            self.forward = self.forwardg1
    def forwarde1(self,x):
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity[...,self.indsidentity],2,0),2,1)*self.identity
        self.cidentity = cidentity
        return torch.nn.functional.conv2d(x,cidentity)
    def forwardg1(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        cxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxx,2,0),2,1)[...,self.indsxx]*self.XhatXhat
        cx90x = torch.repeat_interleave(torch.repeat_interleave(self.weightsx90x,2,0),2,1)[...,self.indsxx]*self.X90hatXhat
        cxx90 = torch.repeat_interleave(torch.repeat_interleave(self.weightsxx90,2,0),2,1)[...,self.indsxx]*self.XhatX90hat
        cx90x90 = torch.repeat_interleave(torch.repeat_interleave(self.weightsx90x90,2,0),2,1)[...,self.indsxx]*self.X90hatX90hat
        cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity,2,0),2,1)[...,self.indsidentity]*self.identity
        c = cxx + cx90x + cxx90 + cx90x90 + cidentity
        self.c = c
        self.cxx = cxx # don't really need this, but may want to look at it later
        self.cidentity = cidentity
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c) # no bias when output is vector
    
class ScalarVectorToScalarVector(torch.nn.Module):
    def __init__(self, in_scalars, in_vectors, out_scalars, out_vectors, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        
        self.in_scalars = in_scalars
        self.in_vectors = in_vectors
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        
        if in_scalars > 0 and out_scalars > 0:
            self.ss = ScalarToScalar(in_scalars, out_scalars, kernel_size, padding, bias, padding_mode)
        if in_scalars > 0 and out_vectors > 0:
            self.sv = ScalarToVector(in_scalars, out_vectors, kernel_size, padding, padding_mode)
        if in_vectors > 0 and out_scalars > 0:
            self.vs = VectorToScalar(in_scalars, out_scalars, kernel_size, padding, bias, padding_mode)
        if in_vectors > 0 and out_vectors > 0:
            self.vv = VectorToVector(in_scalars, out_vectors, kernel_size, padding, padding_mode)
            
        # it seems there are 16 total possibilities for forward functions given missing data
        # perhaps we could handle these cases above?
        
        
    def forward(self,x):
        # TODO implement this without if statements
        outs = torch.zeros((x.shape[0],self.out_scalars,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        outv = torch.zeros((x.shape[0],self.out_vectors*2,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        #print(outs.shape,outv.shape)
        if self.in_scalars > 0 and self.out_scalars > 0:
            outs = outs + self.ss( x[:,:self.in_scalars])
        if self.in_scalars > 0 and self.out_vectors > 0:
            outv = outv + self.sv( x[:,:self.in_scalars])
        if self.in_vectors > 0 and self.out_scalars > 0:
            outs = outs + self.vs(x[:,self.in_scalars:])
        if self.in_vectors > 0 and self.out_vectors > 0:
            outv = outv + self.vv(x[:,self.in_scalars:])        
        #print(outs.shape,outv.shape)
        
        return torch.concatenate(  ( outs, outv )  , dim=-3)
    
    
class Downsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,x):
        # downsample on the last two dimensions by a factor of 2
        # if it is even, we average
        # if it is odd we skip
        if not x.shape[-1]%2: # if even
            x = (x[...,0::2] + x[...,1::2])*0.5
        else:
            x = x[...,0::2]
        
        if not x.shape[-2]%2: # if even
            x = (x[...,0::2,:] + x[...,1::2,:])*0.5
        else:
            x = x[...,0::2,:]
        
        return x
            
    
class Upsample(torch.nn.Module):
    def __init__(self):
        super().__init__()        
    def forward(self,x,roweven=True,coleven=True):
        
        if coleven:
            x = torch.repeat_interleave(x,2,dim=-1)
        else:
            # if odd we insert zeros
            x = (torch.repeat_interleave(x,2,dim=-1) * (1-torch.arange(2*x.shape[-1])%2))[...,:-1]
        
        if roweven:
            x = torch.repeat_interleave(x,2,dim=-2)
        else:
            x = (torch.repeat_interleave(x,2,dim=-2) * (1-torch.arange(2*x.shape[-2])%2)[...,None])[...,:-1,:]
        return x

    
    
# as before, the sigmoid is causing a problem that leads to nans
# perhaps because the sqrt has an infinite slope at x=0?
class VectorSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #return torch.relu(x)
        #return torch.abs(x)
        # the vector has some multiple of 2 chanels
        
        x2 = x**2
        l2 = x2[:,0::2] + x2[:,1::2] + 1e-6
        #l2r = torch.repeat_interleave(l2,2,dim=1)
        #return x * l2r / (l2r + 1.0)
        #return x / torch.sqrt((l2r + 1.0))
        #return x*torch.relu(l2r-1)/l2r
        l = torch.sqrt(l2)
        # now I have the length of each vector
        lr = torch.repeat_interleave(l,2,dim=1)
        # now it is repeated
        return x*torch.relu((lr-1.0))/lr
        
        #return torch.relu(x)
class VectorSigmoidLog(torch.nn.Module):
    '''This one is just relu on the log magnitude'''
    def __init__(self,ep=1e-6):
        super().__init__()
        self.ep = ep
    def forward(self,x):
        # first get magnitude
        x2 = x**2
        l2 = x2[:,0::2] + x2[:,1::2] + self.ep
        logl2 = torch.log(l2)
        newlogl2 = torch.relu(logl2)
        factor = ( (newlogl2 - logl2)*0.5 ).exp()
        return x*factor.repeat_interleave(2,dim=1)
        
        
        
class ScalarSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #l = torch.sqrt(x**2 + 1e-5)
        #return x*torch.relu((l-1.0))/l
        return torch.relu(x)
        
class ScalarVectorSigmoid(torch.nn.Module):
    def __init__(self,n_scalars):
        super().__init__()
        self.n_scalars = n_scalars
        self.s = ScalarSigmoid()
        self.v = VectorSigmoid()
    def forward(self,x):
        return torch.concatenate((self.s(x[:,:self.n_scalars]), self.v(x[:,self.n_scalars:])),-3)
        
        
class ScalarBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):
        return self.b(x)
        
        
class VectorBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):                
        magnitude2 = x[:,0::2]**2 + x[:,1::2]**2 + 1e-6
        logmagnitude2 = torch.log(magnitude2)
        #scaledlogmagnitude2 = self.b(logmagnitude2)
        # let's think about this normalization
        # do I really need the 0.5 below?
        
        #return x * torch.repeat_interleave((  (scaledlogmagnitude2 - logmagnitude2)*0.5 ).exp(),2,dim=1)

        logmagnitude = 0.5*torch.log(magnitude2)
        scaledlogmagnitude = self.b(logmagnitude)
        return x * torch.repeat_interleave((  (scaledlogmagnitude - logmagnitude) ).exp(),2,dim=1)
        
class ScalarVectorBatchnorm(torch.nn.Module):
    def __init__(self,nscalar,nvector):
        super().__init__()
        self.nscalar = nscalar
        self.nvector = nvector
        self.bs = ScalarBatchnorm(nscalar)
        self.bv = VectorBatchnorm(nvector)
    def forward(self,x):
        return torch.concatenate( (self.bs(x[:,:self.nscalar]),self.bv(x[:,self.nscalar:])) , 1)
    
    
    
class ScalarToMatrix(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        super().__init__()
        # what's the main idea here?
        # for the identity, we can do a regular conv, then multiply by identity
        # for the xx we have to actually do the bigger convolution
        # since we're doing the bigger convolution, we might as well just do it
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)
        XhatXhat = Xhat[None,:]*Xhat[:,None] # 2x2xkxk
        XhatXhat = XhatXhat.reshape(4,1,kernel_size,kernel_size) # 4x1xkxk
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).reshape(4,1).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
    def forward(self,x):
        # note
        # the input is going to have in_channels
        # the output is going to have out_channels*4              
        cxx = torch.repeat_interleave(self.weightsxx,4,0)[...,self.indsxx] * self.XhatXhat
        cidentity = torch.repeat_interleave(self.weightsidentity,4,0)[...,self.indsidentity]*self.identity
        c = cxx + cidentity        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c) # no bias when output is matrix
                
        
        
class MatrixToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        super().__init__()
        # what's the main idea here?
        # for the identity, we can do a regular conv, then multiply by identity
        # for the xx we have to actually do the bigger convolution
        # since we're doing the bigger convolution, we might as well just do it
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)
        XhatXhat = Xhat[None,:]*Xhat[:,None]      
        XhatXhat = XhatXhat.reshape(1,4,kernel_size,kernel_size)
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).reshape(1,4).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))       
        
    def forward(self,x):                   
        cxx = torch.repeat_interleave(self.weightsxx,4,1)[...,self.indsxx]*self.XhatXhat
        cidentity = torch.repeat_interleave(self.weightsidentity,4,1)[...,self.indsidentity]*self.identity
        c = cxx + cidentity        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c,self.bias) 
    
class MatrixToVector(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        ''' Here we need to act with an operator that has 3 indices, and sum over two of them
        '''
        super().__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxxx = inds.clone()-1
        indsxxx[indsxxx==-1] = 0 # will get multiplied by zero        
        indsidentity = inds
        # identity
        identity = torch.eye(2)[:,:,None,None]
        # build up Xhat
        Xhat = X/R[...,None]
        Xhat[R==0] = 0  
        Xhat = Xhat.permute(-1,0,1) # put the vector components in the front
        
        # now we have these guys
        XXX = Xhat[:,None,None]*Xhat[None,:,None]*Xhat[None,None,:]
        # or
        XDD = Xhat[:,None,None] * identity[None,:,:]
        # or
        DXD = Xhat[None,:,None]*identity[:,None,:]
        # or
        DDX = identity[:,:,None]*Xhat[None,None,:]
        # now reshape them and tile them
        XXX = XXX.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDD = XDD.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXD = DXD.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDX = DDX.reshape(2,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
                                                           
        
        
        # register buffers, this will allow them to move to devices        
        self.register_buffer('XXX',XXX)
        self.register_buffer('XDD',XDD)
        self.register_buffer('DXD',DXD)
        self.register_buffer('DDX',DDX)
        self.register_buffer('indsxxx',indsxxx)
                
        self.weightsxxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxdd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsddx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        
        
    def forward(self,x):                           
        cxxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxx,4,1),2,0)[...,self.indsxxx]*self.XXX
        cddx = torch.repeat_interleave(torch.repeat_interleave(self.weightsddx,4,1),2,0)[...,self.indsxxx]*self.DDX
        cdxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxd,4,1),2,0)[...,self.indsxxx]*self.DXD
        cxdd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxdd,4,1),2,0)[...,self.indsxxx]*self.XDD
        
        
        c = cxxx + cddx + cdxd + cxdd        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c)     

class VectorToMatrix(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        ''' Here we need to act with an operator that has 3 indices, and sum over two of them
        '''
        super().__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxxx = inds.clone()-1
        indsxxx[indsxxx==-1] = 0 # will get multiplied by zero        
        indsidentity = inds
        # identity
        identity = torch.eye(2)[:,:,None,None]
        # build up Xhat
        Xhat = X/R[...,None]
        Xhat[R==0] = 0  
        Xhat = Xhat.permute(-1,0,1) # put the vector components in the front
        
        # now we have these guys
        XXX = Xhat[:,None,None]*Xhat[None,:,None]*Xhat[None,None,:]
        # or
        XDD = Xhat[:,None,None] * identity[None,:,:]
        # or
        DXD = Xhat[None,:,None]*identity[:,None,:]
        # or
        DDX = identity[:,:,None]*Xhat[None,None,:]
        # now reshape them and tile them
        XXX = XXX.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDD = XDD.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXD = DXD.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDX = DDX.reshape(4,2,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
           
        
        
        
        
        
        
        
        
        # register buffers, this will allow them to move to devices        
        self.register_buffer('XXX',XXX)
        self.register_buffer('XDD',XDD)
        self.register_buffer('DXD',DXD)
        self.register_buffer('DDX',DDX)
        self.register_buffer('indsxxx',indsxxx)
        
        
        self.weightsxxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxdd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsddx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        
        
    def forward(self,x):                           
        cxxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxx,2,1),4,0)[...,self.indsxxx]*self.XXX
        cddx = torch.repeat_interleave(torch.repeat_interleave(self.weightsddx,2,1),4,0)[...,self.indsxxx]*self.DDX
        cdxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxd,2,1),4,0)[...,self.indsxxx]*self.DXD
        cxdd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxdd,2,1),4,0)[...,self.indsxxx]*self.XDD
        
        
        c = cxxx + cddx + cdxd + cxdd        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c)     

    
class MatrixToMatrix(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True,padding_mode='zeros'):
        ''' Here we need to act with an operator that has 3 indices, and sum over two of them
        '''
        super().__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxxxx = inds.clone()-1
        indsxxxx[indsxxxx==-1] = 0 # will get multiplied by zero        
        indsidentity = inds
        # identity
        identity = torch.eye(2)[:,:,None,None]
        # build up Xhat
        Xhat = X/R[...,None]
        Xhat[R==0] = 0  
        Xhat = Xhat.permute(-1,0,1) # put the vector components in the front
        
        # first all Xs (1)
        XXXX = Xhat[:,None,None,None]*Xhat[None,:,None,None]*Xhat[None,None,:,None]*Xhat[None,None,None,:]
        # now with one identity (6)         
        XXDD = Xhat[:,None,None,None]*Xhat[None,:,None,None]*identity[None,None,:,:]
        # or
        XDXD = Xhat[:,None,None,None]*identity[None,:,None,:]*Xhat[None,None,:,None]
        # or
        XDDX = Xhat[:,None,None,None]*identity[None,:,:,None]*Xhat[None,None,None,:]
        # or
        DXXD = identity[:,None,None,:]*Xhat[None,:,None,None]*Xhat[None,None,:,None]
        # or
        DXDX = identity[:,None,:,None]*Xhat[None,:,None,None]*Xhat[None,None,None,:]
        # or
        DDXX = identity[:,:,None,None]*Xhat[None,None,:,None]*Xhat[None,None,None,:]
        # now with two identities (2)
        DDDD0 = identity[:,:,None,None]*identity[None,None,:,:]
        DDDD1 = identity[:,None,:,None]*identity[None,:,None,:]
        DDDD2 = identity[:,None,None,:]*identity[None,:,:,None]
        
        
        # now reshape them and tile them        
        XXXX = XXXX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XXDD = XXDD.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDXD = XDXD.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        XDDX = XDDX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXXD = DXXD.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DXDX = DXDX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDXX = DDXX.reshape(4,4,kernel_size,kernel_size).repeat(out_channels,in_channels,1,1)
        DDDD0 = DDDD0.reshape(4,4,1,1).repeat(out_channels,in_channels,1,1)
        DDDD1 = DDDD1.reshape(4,4,1,1).repeat(out_channels,in_channels,1,1)
        DDDD2 = DDDD2.reshape(4,4,1,1).repeat(out_channels,in_channels,1,1)
        

        
        
        
        # register buffers, this will allow them to move to devices        
        self.register_buffer('XXXX',XXXX)
        self.register_buffer('XXDD',XXDD)        
        self.register_buffer('XDXD',XDXD)
        self.register_buffer('XDDX',XDDX)
        self.register_buffer('DXXD',DXXD)
        self.register_buffer('DXDX',DXDX)
        self.register_buffer('DDXX',DDXX)
        self.register_buffer('DDDD0',DDDD0)
        self.register_buffer('DDDD1',DDDD1)
        self.register_buffer('DDDD2',DDDD2)
        
        self.register_buffer('indsxxxx',indsxxxx)
        
        self.register_buffer('indsidentity',indsidentity)
        
        
        self.weightsxxxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxxdd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxdxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsxddx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxxd = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdxdx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsddxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsdddd0 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        self.weightsdddd1 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        self.weightsdddd2 = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
        
        
        
    def forward(self,x):                           
        cxxxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxxx,4,1),4,0)[...,self.indsxxxx]*self.XXXX
        cxxdd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxxdd,4,1),4,0)[...,self.indsxxxx]*self.XXDD
        cxdxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsxdxd,4,1),4,0)[...,self.indsxxxx]*self.XDXD
        cxddx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxddx,4,1),4,0)[...,self.indsxxxx]*self.XDDX
        cdxxd = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxxd,4,1),4,0)[...,self.indsxxxx]*self.DXXD
        cdxdx = torch.repeat_interleave(torch.repeat_interleave(self.weightsdxdx,4,1),4,0)[...,self.indsxxxx]*self.DXDX
        cddxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsddxx,4,1),4,0)[...,self.indsxxxx]*self.DDXX
        cdddd0 = torch.repeat_interleave(torch.repeat_interleave(self.weightsdddd0,4,1),4,0)[...,self.indsidentity]*self.DDDD0
        cdddd1 = torch.repeat_interleave(torch.repeat_interleave(self.weightsdddd1,4,1),4,0)[...,self.indsidentity]*self.DDDD1
        cdddd2 = torch.repeat_interleave(torch.repeat_interleave(self.weightsdddd2,4,1),4,0)[...,self.indsidentity]*self.DDDD2
        
        
        c = cxxxx + cxxdd + cxdxd + cxddx + cdxxd + cdxdx + cddxx + cdddd0 + cdddd1 + cdddd2 
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)        
        return torch.nn.functional.conv2d(tmp,c)     
    
class MatrixSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
                        
        x2 = x**2
        l2 = x2[:,0::4] + x2[:,1::4] + x2[:,2::4] + x2[:,3::4] + 1e-6        
        l = torch.sqrt(l2)
        # now I have the length of each vector
        lr = torch.repeat_interleave(l,4,dim=1)
        # now it is repeated
        return x*torch.relu((lr-1.0))/lr
                
class MatrixBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):                
        magnitude2 = x[:,0::4]**2 + x[:,1::4]**2 + x[:,2::4]**2 + x[:,3::4]**2 + 1e-6
        logmagnitude2 = torch.log(magnitude2)
        scaledlogmagnitude2 = self.b(logmagnitude2)
        
        return x * torch.repeat_interleave((  (scaledlogmagnitude2 - logmagnitude2)*0.5 ).exp(),4,dim=1)
        
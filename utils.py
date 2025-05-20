import numpy as np
from numpy.random import uniform
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from scipy.spatial.distance import directed_hausdorff
# Sphere

def sample_sphere(size):
    N=int(1.6*1e6)

    phi=np.random.uniform(size=size)*2*np.pi

    # Rejection sampling for Latitude
    theta=np.random.uniform(size=N)*np.pi-np.pi/2
    u=np.random.uniform(size=N)
    theta=theta[u<np.cos(theta)][:size]

    return theta,phi

def sample_reeb_sphere(z,size,r):
    theta=np.repeat(np.arcsin(z/r),size)

    phi=np.random.uniform(size=size)*2*np.pi
    
    return theta,phi


def to_cartesian_sphere(theta,phi,r):
    x=r*np.cos(theta)*np.cos(phi)
    y=r*np.cos(theta)*np.sin(phi)
    z=r*np.sin(theta)

    return np.array([x,y,z]).T


def compute_gdc(theta1,phi1,theta2,phi2,r):
    
    ## Haversine formula
    return 2*r*np.arcsin(np.sqrt(np.sin((theta1-theta2)/2)**2+np.cos(theta1)*np.cos(theta2)*np.sin((phi1-phi2)/2)**2))



def mapper(filter,res,gain):
    
    max=filter.max()
    min=filter.min()
    
    column_indices=np.array(range(res)).reshape(1,-1).repeat(filter.shape[0],0)
    
    steps=np.repeat(((max-min)/res),res).reshape(1,-1).repeat(filter.shape[0],0)
    
    mins=np.repeat(min,res).reshape(1,-1).repeat(filter.shape[0],0)
    
    gains=np.repeat(gain,res).reshape(1,-1).repeat(filter.shape[0],0)
    
    epsilons=gains/(2-2*gains)*steps
    
    left_endpoints=mins+steps*column_indices-epsilons
    
    right_endpoints=mins+steps*(column_indices+1)+epsilons
    
    filter_values=filter.reshape((-1,1)).repeat(res,1)
    
    comparison=(filter_values<=right_endpoints)*(filter_values>=left_endpoints)
    
    d={}
    
    for c in range(res):
        d[c]=np.where(comparison[:,c])[0]
    
    for c in range(res):
        if c<res-1:
            inter=np.intersect1d(d[c], d[c+1],assume_unique=True)
            if inter.size!=0:
                d[str([c,c+1])]=inter
                d[c]=np.setdiff1d(d[c], inter, assume_unique=True)
                d[c+1]=np.setdiff1d(d[c+1], inter, assume_unique=True)
    #for i,j in zip(*np.triu_indices(res)):
    #    if i!=j:
    #        inter=np.intersect1d(d[i], d[j])
    #        if inter.size!=0:
    #            d[str([i,j])]=np.intersect1d(d[i], d[j],assume_unique=True)

    return d


def mapper_element_sphere(dd,theta,phi,f,rsize,r,p):
    l=[]
    for v in dd:
        thetar,phir=sample_reeb_sphere(f[v],rsize,r)
        thetam=theta[dd]
        phim=phi[dd]
        dmat=compute_gdc(thetar[0],np.tile(phir,(thetam.shape[0],)),
                 np.repeat(thetam,rsize),np.repeat(phim,rsize),r).reshape((thetam.shape[0],-1))
        l.append(np.max([np.min(dmat,axis=0).max(),np.min(dmat,axis=1).max()]))
    return(np.array(l))

def compute_loss_sphere(n,gain,res,r,p=1,rsize=1000):
    
    theta,phi=sample_sphere(n)
    f=r*np.sin(theta)
    
    d=mapper(f,res,gain)
    l=Parallel(n_jobs=-1)(delayed(mapper_element_sphere)(dd,theta,phi,f,rsize,r,p) for dd in d.values())
    #for dd in d.values():
    #    for v in dd:
    #        reeb=sample_sphere_reeb(ps[v,2],rsize,r)
    #        mpr=ps[dd,:]
    #        dmat=compute_gdc(np.tile(reeb,(mpr.shape[0],1)),np.repeat(mpr,reeb.shape[0],0),r).reshape((mpr.shape[0],-1))
    #        l+=np.max([np.min(dmat,axis=0).max(),np.min(dmat,axis=1).max()])
            #l+=np.min(dmat,axis=1).max()
    return(np.concatenate(l))


# Torus

def sample_torus(size,R,r):
    n=int(size*2)
    
    phi=uniform(size=size)*2*np.pi

    # Rejection sampling for theta
    theta=uniform(size=n)*2*np.pi
    
    u=uniform(size=n)
    
    theta=theta[u<=(R+r*np.cos(theta))/(R+r)]
    theta=theta[:size]

    x=(R+r*np.cos(theta))*np.cos(phi)
    y=(R+r*np.cos(theta))*np.sin(phi)
    z=r*np.sin(theta)
    
    return (np.array([x,y,z]).T)[:size,:]  


def sample_reeb_torus(x,size,R,r):

    n=int(size/4)
    
    left1=0
    right1=np.pi
    
    left2=np.pi
    right2=2*np.pi
    
    if x+1<1:
        left1=np.arccos(x+1)
        right2=2*np.pi-np.arccos(x+1)
    
    if x-1>-1:
        right1=np.arccos(x-1)
        left2=2*np.pi-np.arccos(x-1)
    
    u=uniform(size=n)*(right1-left1)+left1
    
    v=uniform(size=n)*(right2-left2)+left2
    
    phi=np.concatenate((u,v))
    
    costheta=x-np.cos(phi)
    
    theta=np.concatenate((np.arccos(costheta),2*np.pi-np.arccos(costheta)))
    
    phi=np.tile(phi,(2))

    x=(R+r*np.cos(theta))*np.cos(phi)
    y=(R+r*np.cos(theta))*np.sin(phi)
    z=r*np.sin(theta)

    return np.array([x,y,z]).T


def sample_torus_angles(size,R,r):

    n=int(size*2)
    
    phi=uniform(size=size)*2*np.pi

    # Rejection sampling for theta
    theta=uniform(size=n)*2*np.pi
    
    u=uniform(size=n)
    
    theta=theta[u<=(R+r*np.cos(theta))/(R+r)]
    theta=theta[:size]
    
    return (theta,phi)


def mapper_element_torus(dd,ps,rsize,R,r,f):
    l=[]
    for v in dd:
        reeb=sample_reeb_torus(f[v,0],rsize,R,r)
        mpr=ps[dd,:]
        #dmat=np.linalg.norm(np.tile(reeb,(mpr.shape[0],1))-np.repeat(mpr,reeb.shape[0],0),axis=1).reshape((mpr.shape[0],-1))
        #l+=np.max([np.min(dmat,axis=0).max(),np.min(dmat,axis=1).max()])**p
        l.append(np.max((directed_hausdorff(reeb,mpr)[0],directed_hausdorff(mpr,reeb)[0])))
    return(np.array(l))



def compute_loss_torus(n,gain,res,R,r,rsize=1000):
    
    (theta,phi)=sample_torus_angles(n,R,r)
    ps=np.array([(R+r*np.cos(theta))*np.cos(phi),(R+r*np.cos(theta))*np.sin(phi),r*np.sin(theta)]).T

    f=(np.cos(theta)+np.cos(phi)).reshape((-1,1))
    
    d=mapper(np.cos(theta)+np.cos(phi),res,gain)
    l=Parallel(n_jobs=-1)(delayed(mapper_element_torus)(dd,ps,rsize,R,r,f) for dd in d.values())
    #for dd in d.values():
    #    for v in dd:
    #        reeb=sample_sphere_reeb(ps[v,2],rsize,r)
    #        mpr=ps[dd,:]
    #        dmat=compute_gdc(np.tile(reeb,(mpr.shape[0],1)),np.repeat(mpr,reeb.shape[0],0),r).reshape((mpr.shape[0],-1))
    #        l+=np.max([np.min(dmat,axis=0).max(),np.min(dmat,axis=1).max()])
            #l+=np.min(dmat,axis=1).max()
    return(np.concatenate(l))




def sample_reeb_torus2(x0,size,R,r):
    n=size//2
    
    val=np.max(((x0-R)/r,(-x0-R)/r))
    
    b=np.pi
    
    if val>-1:
        b=np.arccos(val)

    theta=uniform(size=n)*2*b-b
    
    cosphi=x0/(R+r*np.cos(theta))
    
    phi=np.concatenate((np.arccos(cosphi),2*np.pi-np.arccos(cosphi)))
    
    theta=np.tile(theta,(2))
    
    x=(R+r*np.cos(theta))*np.cos(phi)
    y=(R+r*np.cos(theta))*np.sin(phi)
    z=r*np.sin(theta)
    
    return np.array([x,y,z]).T



def sample_reeb_torus2_cc(x0,y0,size,R,r):

    n=size//2
    
    if x0<R-r and x0>-R+r:
        n=size

    val=np.max(((x0-R)/r,(-x0-R)/r))
    
    b=np.pi
    
    if val>-1:
        b=np.arccos(val)
    
    theta=uniform(size=n)*2*b-b
    
    cosphi=x0/(R+r*np.cos(theta))
    
    phi=np.concatenate((np.arccos(cosphi),2*np.pi-np.arccos(cosphi)))
    
    theta=np.tile(theta,(2))
    
    x=(R+r*np.cos(theta))*np.cos(phi)
    y=(R+r*np.cos(theta))*np.sin(phi)
    z=r*np.sin(theta)

    if x0<R-r and x0>-R+r:
        points=np.array([x,y,z]).T
        points=points[points[:,1]*y0>0]

        return points
    
    return np.array([x,y,z]).T


def mapper2(filter,res,gain,sample,R,r): 
    max=filter.max()
    min=filter.min()
    
    column_indices=np.array(range(res)).reshape(1,-1).repeat(filter.shape[0],0)
    
    steps=np.repeat(((max-min)/res),res).reshape(1,-1).repeat(filter.shape[0],0)
    
    mins=np.repeat(min,res).reshape(1,-1).repeat(filter.shape[0],0)
    
    gains=np.repeat(gain,res).reshape(1,-1).repeat(filter.shape[0],0)
    
    epsilons=gains/(2-2*gains)*steps
    
    left_endpoints=mins+steps*column_indices-epsilons
    
    right_endpoints=mins+steps*(column_indices+1)+epsilons
    
    filter_values=filter.reshape((-1,1)).repeat(res,1)
    
    comparison=(filter_values<=right_endpoints)*(filter_values>=left_endpoints)
    
    tcc_ind=np.where((left_endpoints[0,:]>-R+r)*(right_endpoints[0,:]<R-r))[0]
    
    d={}
    
    for c in range(res):
        if c in tcc_ind:
            ut=np.where(comparison[:,c])[0]
            d[c]=[ut[np.where(sample[ut,1]>0)[0]],ut[np.where(sample[ut,1]<0)[0]]]
        else:
            d[c]=np.where(comparison[:,c])[0]
    
    #for i,j in zip(*np.triu_indices(res)):
    #    if i!=j:
    #        inter=np.intersect1d(d[i], d[j])
    #        if inter.size!=0:
    #            d[str([i,j])]=np.intersect1d(d[i], d[j],assume_unique=True)
    for c in range(tcc_ind[0]-1):
        inter=np.intersect1d(d[c], d[c+1],assume_unique=True)
        if inter.size!=0:
            d[str([c,c+1])]=inter
            d[c]=np.setdiff1d(d[c], inter, assume_unique=True)
            d[c+1]=np.setdiff1d(d[c+1], inter, assume_unique=True)
    
    c=tcc_ind[0]-1
    for i in range(2):
        inter=np.intersect1d(d[c], d[c+1][i],assume_unique=True)
        if inter.size!=0:
            d[str([c,c+1,i])]=inter
            d[c]=np.setdiff1d(d[c], inter, assume_unique=True)
            d[c+1][i]=np.setdiff1d(d[c+1][i], inter, assume_unique=True)
    
    for c in tcc_ind[:-1]:
        for i in range(2):
            inter=np.intersect1d(d[c][i], d[c+1][i],assume_unique=True)
            if inter.size!=0:
                d[str([c,c+1,i])]=inter
                d[c][i]=np.setdiff1d(d[c][i], inter, assume_unique=True)
                d[c+1][i]=np.setdiff1d(d[c+1][i], inter, assume_unique=True)
    
    c=tcc_ind[-1]
    for i in range(2):
        inter=np.intersect1d(d[c][i], d[c+1],assume_unique=True)
        if inter.size!=0:
            d[str([c,c+1,i])]=inter
            d[c][i]=np.setdiff1d(d[c][i], inter, assume_unique=True)
            d[c+1]=np.setdiff1d(d[c+1], inter, assume_unique=True)
            
    for c in range(tcc_ind[-1]+1,res-1):
        inter=np.intersect1d(d[c], d[c+1],assume_unique=True)
        if inter.size!=0:
            d[str([c,c+1])]=inter
            d[c]=np.setdiff1d(d[c], inter, assume_unique=True)
            d[c+1]=np.setdiff1d(d[c+1], inter, assume_unique=True)

    return d


def mapper_element_torus2(dd,ps,rsize,R,r):
    l=[]
    if type(dd)==list:
        for i in range(2):
            for v in dd[i]:
                reeb=sample_reeb_torus2_cc(ps[v,0],ps[v,1],rsize,R,r)
                mpr=ps[dd[i],:]
                l.append(np.max((directed_hausdorff(reeb,mpr)[0],directed_hausdorff(mpr,reeb)[0])))
    else:
        for v in dd:
            reeb=sample_reeb_torus2_cc(ps[v,0],ps[v,1],rsize,R,r)
            mpr=ps[dd,:]
            l.append(np.max((directed_hausdorff(reeb,mpr)[0],directed_hausdorff(mpr,reeb)[0])))
    return(np.array(l))

def compute_loss_torus2(n,gain,res,R,r,rsize=1000):
    
    ps=sample_torus(n,R,r)
        
    d=mapper2(ps[:,0],res,gain,ps,R,r)
    l=Parallel(n_jobs=-1)(delayed(mapper_element_torus2)(dd,ps,rsize,R,r) for dd in d.values())
    #for dd in d.values():
    #    for v in dd:
    #        reeb=sample_sphere_reeb(ps[v,2],rsize,r)
    #        mpr=ps[dd,:]
    #        dmat=compute_gdc(np.tile(reeb,(mpr.shape[0],1)),np.repeat(mpr,reeb.shape[0],0),r).reshape((mpr.shape[0],-1))
    #        l+=np.max([np.min(dmat,axis=0).max(),np.min(dmat,axis=1).max()])
            #l+=np.min(dmat,axis=1).max()
    return(np.concatenate(l))



import numpy as np
from numpy.random import uniform
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from scipy.spatial.distance import directed_hausdorff

# Get Mapper mm-space parameters

def max_dis(i,j,points,dd,D):
    I=list(dd.keys())[i]
    J=list(dd.keys())[j]
    if dd[I].size==0 or dd[J].size==0:
        return D
    return np.max((directed_hausdorff(points[dd[I],:],points[dd[J],:])[0],directed_hausdorff(points[dd[J],:],points[dd[I],:])[0]))


def get_ot_params(mapper,mes,points,D):

    dd={}
    st=mapper.simplex_tree
    for k in mapper.node_info.keys():
        dd[k]=mapper.node_info[k]['indices']

    for (splx,_) in st.get_skeleton(1):
        if len(splx) == 2:
            dd[str(splx)]=np.intersect1d(dd[splx[0]], dd[splx[1]],assume_unique=True)
            dd[splx[0]]=np.setdiff1d(dd[splx[0]], dd[str(splx)], assume_unique=True)
            dd[splx[1]]=np.setdiff1d(dd[splx[1]], dd[str(splx)], assume_unique=True)

    p=np.array([mes[dd[k]].sum() for k in dd.keys()])
    inds=np.triu_indices(len(dd.keys()))

    #dm=distance_matrix(vertices, vertices)

    out=Parallel(n_jobs=-1)(delayed(max_dis)(i, j,points,dd,D) for (i,j) in zip(*inds))

    C=np.zeros((len(dd.keys()),len(dd.keys())))

    l=list(zip(*inds))
    for k in range(len(l)):
        v=out[k]
        (i,j)=l[k]
        C[i,j]=v
        C[j,i]=v

    return(p,C)

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


def sample_phi(size,alpha,beta):

    u=uniform(size=size)
    v=uniform(size=size)
    
    v[u<alpha]=v[u<alpha]*np.pi/3+5*np.pi/6
    v[(u>alpha)*(u<alpha+beta)]=v[(u>alpha)*(u<alpha+beta)]*np.pi/3-np.pi/6
    v[v<0]=v[v<0]+2*np.pi
    v[(u>alpha+beta)*(u<(1+alpha+beta)/2)]=v[(u>alpha+beta)*(u<(1+alpha+beta)/2)]*2*np.pi/3+np.pi/6
    v[u>(1+alpha+beta)/2]=v[u>(1+alpha+beta)/2]*2*np.pi/3+7*np.pi/6

    return v


def sample_torus_unbalanced(size,R,r,alpha,beta):
    n=int(size*2)
    
    
    
    #phi=uniform(size=size)*np.pi+(uniform(size=size)<1/4)*np.pi
    phi=sample_phi(size,alpha,beta)
    
    # Rejection sampling for theta
    theta=uniform(size=n)*2*np.pi
    
    u=uniform(size=n)
    
    theta=theta[u<=(R+r*np.cos(theta))/(R+r)]
    theta=theta[:size]

    x=(R+r*np.cos(theta))*np.cos(phi)
    y=(R+r*np.cos(theta))*np.sin(phi)
    z=r*np.sin(theta)
    
    return (np.array([x,y,z]).T)[:size,:]  



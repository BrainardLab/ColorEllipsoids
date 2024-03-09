import numpy as np
from ellipsoid_fit import ellipsoid_fit as ellipsoid_fit, data_regularize


if __name__=='__main__':

    data = np.loadtxt("data.txt")
    data2 = data_regularize(data)

    center, radii, evecs, v, chi2 = ellipsoid_fit(data2)

    a,b,c = radii
    r = (a*b*c)**(1./3.)
    D = np.array([[r/a,0.,0.],[0.,r/b,0.],[0.,0.,r/c]])
    TR = evecs.dot(D).dot(evecs.T)
    
    print 'residual:', chi2
    print 'center: ',center
    print 'transformation:'
    print TR
    
    np.savetxt('result.txt', np.vstack((center.T, TR)))


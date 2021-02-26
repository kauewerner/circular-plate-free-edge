"""
This script can be used to obtain the coefficients (lambda_mn and C_mn) to compute the natural frequencies and vibration modes of circular plates with free edge boundary conditions.

It is based on the same approach used for circular plates with elastic edge support described by:
    - Zagrai, A., & Donskoy, D. (2005). A “soft table” for the natural frequencies and modal parameters of uniform circular plates with elastic edge support. Journal of Sound and Vibration, 287(1-2), 343-351.
    - Shared MATLAB code: https://se.mathworks.com/matlabcentral/fileexchange/6474-natural-frequencies-of-a-circular-plate

The code was validated with the results presented in the following papers:
    - Itao, K., & Crandall, S. H. (1979). Natural modes and natural frequencies of uniform, circular, free-edge plates.
    - Amabili, M., Pasqualini, A., & Dalpiaz, G. (1995). Natural frequencies and modes of free-edge circular plates vibrating in vacuum or in contact with liquid. Journal of sound and vibration, 188(5), 685-699.

usage: circular_plate_free_edge.py [-h] [-m ANGULARMODES] [-mr MAXIMUMROOTVALUE] [-v POISSONRATIO] [-d DENSITY] [-ym YOUNGMODULUS]
                                   [-t THICKNESS] [-r RADIUS] [-pf [PLOTFREQUENCIES]] [-ef [EXPORTFREQUENCYRESULTS]] [-o [OUTPUTPATH]]

optional arguments:
  -h, --help            show this help message and exit
  -m ANGULARMODES, --angularModes ANGULARMODES
                        m: maximum number of angular modes. Default = 30
  -mr MAXIMUMROOTVALUE, --maximumRootValue MAXIMUMROOTVALUE
                        Maximum value within the root computation range, from zero to this value. It relates to the maximum number in the
                        range (starting above 0) where the roots (related to radial modes) can be found. Default = 100
  -v POISSONRATIO, --poissonRatio POISSONRATIO
                        Poisson ratio of the plate material. Default = 0.33
  -d DENSITY, --density DENSITY
                        Density of the plate material. Default = 8000
  -ym YOUNGMODULUS, --youngModulus YOUNGMODULUS
                        Youngs Modulus of the plate material. Default = 100e9
  -t THICKNESS, --thickness THICKNESS
                        Thickness of the circular plate. Default = 0.001
  -r RADIUS, --radius RADIUS
                        Radius of the circular plate. Default = 0.25
  -pf [PLOTFREQUENCIES], --plotFrequencies [PLOTFREQUENCIES]
                        Plot natural frequencies.
  -ef [EXPORTFREQUENCYRESULTS], --exportFrequencyResults [EXPORTFREQUENCYRESULTS]
                        Export natural frequencies results.
  -o [OUTPUTPATH], --outputPath [OUTPUTPATH]
                        Export natural frequencies results. Default = '.'

By Kaue Werner, 2021
"""
import os
import sys
import argparse
from scipy import optimize
from scipy import special as sp
import matplotlib.pyplot as plt
import numpy as np

def eigenvalue_function(lb,m,v):
    C1_num = (lb**2)*sp.jv(m,lb) + (1 - v)*(lb*sp.jvp(m,lb,1) - (m**2)*sp.jv(m,lb))
    C1_den = (lb**2)*sp.iv(m,lb) - (1 - v)*(lb*sp.ivp(m,lb,1) - (m**2)*sp.iv(m,lb))
    C2_num = (lb**3)*sp.jvp(m,lb,1) + (m**2)*(1 - v)*(lb*sp.jvp(m,lb,1) - sp.jv(m,lb))
    C2_den = (lb**3)*sp.ivp(m,lb,1) - (m**2)*(1 - v)*(lb*sp.ivp(m,lb,1) - sp.iv(m,lb))
    return ((C1_num/C1_den) - (C2_num/C2_den))

def compute_squared_lambda(M,maxValue,v):
    step = 1 
    roots = np.zeros((M,maxValue))
    squaredLambda = np.zeros((maxValue,M))
    for m in range(M):
        idx = 0 
        for n in range(maxValue):
            a = 1e-32 + step*(n)
            b = step*(n+1)

            fa = eigenvalue_function(a,m,v)
            fb = eigenvalue_function(b,m,v)
            if (fa > 0) == (fb < 0):
                sol = optimize.root_scalar(eigenvalue_function,args=(m,v),method='brentq',bracket=[a,b])
                if (sol.root > m):
                    roots[m,idx] = sol.root
                    idx += 1
        
        print('========================================================\n')
        print('optimize.root_scalar completed for m = ' + str(m)+' with the following n roots:\n')
        temp = roots[m,np.nonzero(roots[m,:])]
        if m < 2:
            initIdx = 1
        else:
            initIdx = 0
        
        for tempIdx in range(len(temp[0,:])): 
            squaredLambda[initIdx + tempIdx,m] = temp[0,tempIdx]**2
        print(temp)
        print('========================================================\n')
    return squaredLambda


def compute_natural_frequencies(squaredLambda,E=100e9,r=0.25,h=0.005,v=0.33,rho=8000,export=False,outputPath='.'):
    D = (E*(h**3))/(12.0*(1.0-(v**2)))
    freqWeight = (1.0/(2.0*np.pi*(r**2)))*np.sqrt(D/(rho*h))
    naturalFrequencies = np.zeros((np.size(squaredLambda,0),np.size(squaredLambda,1)))
    for pidx in range((np.size(squaredLambda,0))):
        naturalFrequencies[pidx,:] = freqWeight*squaredLambda[pidx,:]
    if export:
        outputFile = open(outputPath+'/frequency_results_v='+str(v)+'.dat',"wt")
        outputFile.write("m\tn\tnatural_frequency\n")
        for m in range(np.size(naturalFrequencies,1)):
            temp = naturalFrequencies[np.nonzero(naturalFrequencies[:,m]),m]
            for n in range(len(temp[0,:])):
                if m < 2:
                    idx = n + 1
                else:
                    idx = n
                outputFile.write(str(m)+"\t"+str(idx)+"\t"+str(temp[0,n])+"\n")
    return naturalFrequencies

def plot_frequencies(freqs): 
    for pidx in range(np.size(freqs,0)):
        temp = freqs[pidx,np.nonzero(freqs[pidx,:])]
        plt.plot(np.arange(0,len(temp[0,:])),temp[0,:],'o-')
    plt.xlim(0,30)
    plt.ylim(0,5000)
    plt.show()
    return 0

def compute_C(lb,m,v):
    C1_num = (lb**2)*sp.jv(m,lb) + (1 - v)*(lb*sp.jvp(m,lb,1) - (m**2)*sp.jv(m,lb))
    C1_den = (lb**2)*sp.iv(m,lb) - (1 - v)*(lb*sp.ivp(m,lb,1) - (m**2)*sp.iv(m,lb))
    return (C1_num/C1_den)

def export_results(squaredLambda,v,outputPath='./'):
    outputFile = open(outputPath+'/lambda_results_v='+str(v)+'.dat',"wt")
    outputFile.write("m\tn\tlambda\tsq_lambda\tC\n")
    for m in range(np.size(squaredLambda,1)):
        temp = squaredLambda[np.nonzero(squaredLambda[:,m]),m]
        for n in range(len(temp[0,:])):
            if m < 2:
                idx = n + 1
            else:
                idx = n
            outputFile.write(str(m)+"\t"+str(idx)+"\t"+str(np.sqrt(temp[0,n]))+"\t"+str(temp[0,n])+"\t"+str(compute_C(np.sqrt(temp[0,n]),m,v))+"\n")
    return 0

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--angularModes", help="m: maximum number of angular modes. Default = 30",type=int)
    parser.add_argument("-mr","--maximumRootValue", help="Maximum value within the root computation range, from zero to this value. It relates to the maximum number in the range (starting above 0) where the roots (related to radial modes) can be found. Default = 100",type=int)
    parser.add_argument("-v", "--poissonRatio", help="Poisson ratio of the plate material. Default = 0.33", type=float)
    parser.add_argument("-d", "--density", help="Density of the plate material. Default = 8000", type=float)
    parser.add_argument("-ym", "--youngModulus", help="Youngs Modulus of the plate material. Default = 100e9", type=float)
    parser.add_argument("-t", "--thickness", help="Thickness of the circular plate. Default = 0.001", type=float)
    parser.add_argument("-r", "--radius", help="Radius of the circular plate. Default = 0.25", type=float)
    parser.add_argument("-pf", "--plotFrequencies",nargs='?', help="Plot natural frequencies.", const=True , type=bool)
    parser.add_argument("-ef", "--exportFrequencyResults",nargs='?', help="Export natural frequencies results.", const=True , type=bool)
    parser.add_argument("-o", "--outputPath",nargs='?', help="Export natural frequencies results. Default = '.'", type=str)
    args = parser.parse_args()
    
    M = 30 if not args.angularModes else args.angularModes
    maxValue = 100 if not args.maximumRootValue else args.maximumRootValue
    v = 0.33 if not args.poissonRatio else args.poissonRatio
    plotFlag = False if not args.plotFrequencies else args.plotFrequencies
    h = 0.001 if not args.thickness else args.thickness
    r = 0.25 if not args.radius else args.radius
    rho = 8000.0 if not args.density else args.density
    E = 100e9 if not args.youngModulus else args.youngModulus
    exportFlag = False if not args.exportFrequencyResults else args.exportFrequencyResults
    outputPath = '.' if not args.outputPath else args.outputPath
    
    squaredLambda = compute_squared_lambda(M,maxValue,v)
    export_results(squaredLambda,v,outputPath)
    
    if exportFlag:
        freqs = compute_natural_frequencies(squaredLambda,E,r,h,v,rho,exportFlag,outputPath)

    if plotFlag:
        freqs = compute_natural_frequencies(squaredLambda,E,r,h,v,rho,exportFlag,outputPath)
        plot_frequencies(freqs)

if __name__ == "__main__":
    main(sys.argv[1:])

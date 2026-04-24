import numpy as np
import random
import math
from itertools import combinations
import sys
# np.random.seed(1)
# random.seed(2)
def int_matrix (bbsol,bsloc,bssg):
    chi=np.zeros((5+len(bsloc)*2,5+len(bsloc)*2))
    
    bb1=0 #backbone beads
    sg1=1 #side group

    bb2=2 #backbone beads
    sg2=3 #side group
    
    sol=4 #solvent
    
    chi[:,sol]=bbsol
    chi[sol,:]=bbsol
    
    chi[sol,sol]=0
    
    for i in range (len(bsloc)):
        chi[6+i-1,6+i+len(bsloc)-1]=-6
        chi[6+i+len(bsloc)-1,6+i-1]=-6
        chi[6+i-1,sg1]=bssg
        chi[sg1,6+i-1]=bssg
        chi[6+i-1,sg2]=bssg
        chi[sg2,6+i-1]=bssg
        chi[6+i+len(bsloc)-1,sg1]=bssg
        chi[sg1,6+i+len(bsloc)-1]=bssg
        chi[6+i+len(bsloc)-1,sg2]=bssg        
        chi[sg2,6+i+len(bsloc)-1]=bssg        
    # print(chi[5,6])
    return (chi)

def overlap_matrix (test):
    
    over=np.zeros((4,4))
    
    bb=0 #backbone beads
    bs=2 #binding site
    sg=1 #side group
    sol=3 #solvent
    
    over[bb,bb]=0
    over[bb,bs]=0
    over[bb,sg]=0
    over[bb,sol]=1
    
    over[bs,bb]=over[bb,bs]
    over[bs,bs]=0
    over[bs,sg]=0
    over[bs,sol]=1    
    
    over[sg,bb]=over[bb,sg]
    over[sg,bs]=0
    over[sg,sg]=0
    over[sg,sol]=1 
    
    over[sol,bb]=over[bb,sol]
    over[sol,bs]=over[bs,sol]
    over[sol,sg]=over[sg,sol]
    over[sol,sol]=1

    return (over)    
    

##initial configuration generator##
def initial (lx,ly,N,rho,dL,A,nsol,bsloc):
 
    config=np.zeros((2*N+nsol,5))
    
    count=0
    config[count]=[count+1, 1, 1, 0, 0]
    for i in range (N-2):
        count=count+1
        config[count]=[count+1, 1, 1, 0, i+1]        
    count=count+1
    config[count]=[count+1, 1, 2, 0, i+2]
    for i in range (len(bsloc)):
        config[bsloc[i],2]=6+i
    count=count+1
    config[count]=[count+1, 2, 3, 2, 0]
    for i in range (N-2):
        count=count+1
        config[count]=[count+1, 2, 3, 2, i+1]        
    count=count+1
    config[count]=[count+1, 2, 4, 2, i+2]
    for i in range (len(bsloc)):
        config[bsloc[i]+N,2]=6+i+len(bsloc)    
    # config[bsloc+N,2]=4
    
    
    count2=0    
    
    for i in range (3,lx):
        for j in range (ly):
            if count2<nsol:
            
                count=count+1
                count2=count2+1
                
                config[count]=[count+1, 2+count2, 5, i, j]
    return(config)
    

#function to restart the simulation#
def restart (N):
    file=open('fc_%d_cont_2.txt'%N,'r')
    num_lines=sum(1 for line in open('fc_%d_cont_2.txt'%N))
    config=np.zeros((num_lines,5))
    for i in range (num_lines):
        line=file.readline().split()
        config[i,0]=int(line[0])
        config[i,1]=int(line[1])
        config[i,2]=int(line[2])
        config[i,3]=float(line[3])
        config[i,4]=float(line[4])
    return(config)

#writing configurations in lammps format#

def print_config (config,atoms,configurations,counter):
    configurations.write('ITEM: TIMESTEP\n')
    configurations.write('%d\n'%counter)
    configurations.write('ITEM: NUMBER OF ATOMS\n')
    configurations.write('%d\n'%atoms)
    configurations.write('ITEM: BOX BOUNDS pp pp ff\n')
    configurations.write('0 30\n')
    configurations.write('0 30\n')
    configurations.write('-0.5 0.5\n')
    configurations.write('ITEM: ATOMS id mol type x y z\n')
    for i in range (len(config)):
        configurations.write('%d %d %d %f %f %f\n'%(config[i,0],config[i,1],config[i,2],config[i,3],config[i,4],0))   


#non-bonded energy calculator#
def nbHamiltonian (config,chiact,over,N,lx,ly):
    
    Hnb3=0
    lattice=np.zeros((lx,ly))
   
    for i in range (len(config)):
        lattice[int(config[i,3]),int(config[i,4])]=int(config[i,2])


    lattice = lattice.astype(int)
    loopinglist=config[:2*N,3:5].astype(int)
    if np.count_nonzero(lattice)==len(config):
        # chi1[:,sol]=chi1[:,sol]*2
        # chi1[sol,:]=chi1[sol,:]*2
        for i in range (len(loopinglist)):
            for k in range (-1,2):
                for l in range (-1,2):    
                    if k!=0 or l!=0:
                        if lattice[int(loopinglist[i,0]+k-lx*round((loopinglist[i,0]+k)/lx)),int(loopinglist[i,1]+l-ly*round((loopinglist[i,1]+l)/ly))] !=0:
                            Hnb3=Hnb3+chiact[lattice[loopinglist[i,0],loopinglist[i,1]]-1,lattice[int(loopinglist[i,0]+k-lx*round((loopinglist[i,0]+k)/lx)),int(loopinglist[i,1]+l-ly*round((loopinglist[i,1]+l)/ly))]-1]     

    else:
        Hnb3=10e7
    
    return(Hnb3/2)

#bonded energy calculator#
def bondHamiltonian (config, lx, ly, dL, N, M):
    Hbond3=0
    for k in range (M):
        for j in range (k*N+1,N+k*N):
            Hbond3=Hbond3+(((((config[j,3]-config[j-1,3]-(lx*dL)*round((config[j,3]-config[j-1,3])/(lx*dL),0)))**2+\
            ((config[j,4]-config[j-1,4]-(ly*dL)*round((config[j,4]-config[j-1,4])/(ly*dL),0)))**2)**0.5)-1)**2

    return(Hbond3*15)

#compute ened-to-end distance of the chains#
def e2e (config, N, M, lx, ly,dL):
    re2e=[]
    for k in range (M):
        re2e.append((config[k*N+N-1,3]-config[k*N,3]-(lx*dL)*round((config[k*N+N-1,3]-config[k*N,3])/(lx*dL),0))**2+(config[k*N+N-1,4]-config[k*N,4]-(ly*dL)*round((config[k*N+N-1,4]-config[k*N,4])/(ly*dL),0))**2)
    return(re2e)


#reptation move of the chains#
def rept_move (config, N, lx, ly, dL):
    config1=config.copy()
    moltype=random.choice([0,1])
    direct=random.choice([0,1])
    lenth=random.choice([1,2,3])
    if direct==0:
        for i in range (moltype*N,(moltype+1)*N-lenth):
            config1[i,3:5]=config1[i+lenth,3:5]
        newx=random.choice([-1,0, 1])
        newy=random.choice([-1,0, 1])        
        config1[(moltype+1)*N-lenth:(moltype+1)*N,3]=config1[(moltype+1)*N-lenth:(moltype+1)*N,3]+newx-np.floor((config1[(moltype+1)*N-lenth:(moltype+1)*N,3]+newx)/(lx*dL))*(lx*dL)
        config1[(moltype+1)*N-lenth:(moltype+1)*N,4]=config1[(moltype+1)*N-lenth:(moltype+1)*N,4]+newy-np.floor((config1[(moltype+1)*N-lenth:(moltype+1)*N,4]+newy)/(ly*dL))*(ly*dL)
        
    if direct==1:
        for i in range ((moltype+1)*N-1,moltype*N+lenth-1,-1):
            config1[i,3:5]=config1[i-lenth,3:5]    
            
        newx=random.choice([-1,0, 1])
        newy=random.choice([-1,0, 1])       
        config1[moltype*N:moltype*N+lenth,3]=config1[moltype*N:moltype*N+lenth,3]+newx-np.floor((config1[moltype*N:moltype*N+lenth,3]+newx)/(lx*dL))*(lx*dL)
        config1[moltype*N:moltype*N+lenth,4]=config1[moltype*N:moltype*N+lenth,4]+newy-np.floor((config1[moltype*N:moltype*N+lenth,4]+newy)/(ly*dL))*(ly*dL)
    return(config1)


#local displacement move of hte chains#

def displace_move (config, lx, ly, N, nsol, dL):
    config1=config.copy()
    moltype=random.choices([0,1,2],weights=(10,10,10), k=1)
    
    if moltype[0]==0:
        rows_id=random.sample(range(0, N),1)
        
    elif moltype[0]==1:
        rows_id=random.sample(range(N, N+N),1)
        
    elif moltype[0]==2:
        rows_id=random.sample(range(N, N+nsol),1)
        
    newx=random.choice([-1,0, 1])
    newy=random.choice([-1,0, 1])        
    config1[rows_id,3]=config1[rows_id,3]+newx-math.floor((config1[rows_id,3]+newx)/(lx*dL))*(lx*dL)
    config1[rows_id,4]=config1[rows_id,4]+newy-math.floor((config1[rows_id,4]+newy)/(ly*dL))*(ly*dL)
    return(config1)

#translational movel of solvent molecules#   
def trans_move (config, N, nsol, lx, ly, dL):
    config1=config.copy()
    rows_id=random.sample(range(N, N+nsol),1) 

    config1[rows_id,3]=random.randint(0, lx-1)
    config1[rows_id,4]=random.randint(0, ly-1)        
    return(config1)


#swap chain with cluster of solvent molecules#
def swap (config,N,lx,ly,dL):
    config1=config.copy()
    moltype=random.choice([0,1])
    
    dx=random.randint(-int(lx/2), int(lx/2))  
    dy=random.randint(-int(lx/2), int(ly/2))  
    
    config1[moltype*N:(moltype+1)*N,3:5]=config1[moltype*N:(moltype+1)*N,3:5]+np.transpose(np.array([dx,dy]))-\
    np.floor((config1[moltype*N:(moltype+1)*N,3:5]+np.transpose(np.array([dx,dy])))/(lx*dL))*(lx*dL)

    loopinglist=config1[moltype*N:(moltype+1)*N,3:5].astype(int) #new positions for polymer
    lattice=np.zeros((lx,ly)) # id for atoms in old config
    for i in range (len(config)):
        lattice[int(config[i,3]),int(config[i,4])]=int(config[i,0])
    
    loopinglist1=config[moltype*N:(moltype+1)*N,0].astype(int) #old positions for polymer

    lattice=lattice.astype(int)
    
    for i in range (len(loopinglist)):
        if lattice[loopinglist[i,0],loopinglist[i,1]]>0:
            config1[int(lattice[loopinglist[i,0],loopinglist[i,1]]-1),3:5]=config[loopinglist1[i]-1,3:5]
        
    return(config1)

#association dissociation move#
def assoc_dissoc (config,N,lx,ly,dL,indexes,bsloc,pbias):
    config1=config.copy()
    indexes1=indexes.copy()

    moltype=random.choice([0,1])
    bsid=random.choice([bsloc[i]+moltype*N for i in range (len(bsloc))])

    if moltype==0:
        compbsid=bsid+N
    else:
        compbsid=bsid-N  
        
    bondingmove=random.choices([0,1],weights=[1-pbias,pbias], k=1)[0]
    
    dist_indexes_compbs=np.sum((indexes1-config1[compbsid,3:5])**2,axis=1)**0.5  
    
    if bondingmove==0:
        newbsind=random.choice(np.where(dist_indexes_compbs>2**0.5)[0])
        newbspos=indexes1[int(newbsind)]

        dx=newbspos[0]-config1[bsid,3]
        dy=newbspos[1]-config1[bsid,4]  
    else:
        newbsind=random.choice(np.where((dist_indexes_compbs<=2**0.5)&(dist_indexes_compbs>0))[0])
        newbspos=indexes1[int(newbsind)]

        dx=newbspos[0]-config1[bsid,3]
        dy=newbspos[1]-config1[bsid,4]  
        
    config1[moltype*N:(moltype+1)*N,3:5]=config1[moltype*N:(moltype+1)*N,3:5]+np.transpose(np.array([dx,dy]))-\
    np.floor((config1[moltype*N:(moltype+1)*N,3:5]+np.transpose(np.array([dx,dy])))/(lx*dL))*(lx*dL)

    loopinglist=config1[moltype*N:(moltype+1)*N,3:5].astype(int) #new positions for polymer
    lattice=np.zeros((lx,ly)) # id for atoms in old config
    for i in range (len(config)):
        lattice[int(config[i,3]),int(config[i,4])]=int(config[i,0])
    
    loopinglist1=config[moltype*N:(moltype+1)*N,0].astype(int) #old positions for polymer

    lattice=lattice.astype(int)
    
    for i in range (len(loopinglist)):
        if lattice[loopinglist[i,0],loopinglist[i,1]]>0:
            config1[int(lattice[loopinglist[i,0],loopinglist[i,1]]-1),3:5]=config[loopinglist1[i]-1,3:5]
        
    return(config1,bsid,compbsid)


#compute binding sites distance#

def connectivity (config,bsloc,N,lx,ly,dL):
    bb1=0 #backbone beads
    sg1=1 #side group

    bb2=2 #backbone beads
    sg2=3 #side group
    
    sol=4 #solvent
    
    num_lst=[]
    distances=[]
    op=0
    bonding=[]
    for i in range (len(bsloc)):
        op=op+((config[bsloc[i],3]-config[bsloc[i]+N,3]-(lx*dL)*round((config[bsloc[i],3]-config[bsloc[i]+N,3])/(lx*dL),0))**2+(config[bsloc[i],4]-config[bsloc[i]+N,4]-(lx*dL)*round((config[bsloc[i],4]-config[bsloc[i]+N,4])/(lx*dL),0))**2)
        bonding.append(((config[bsloc[i],3]-config[bsloc[i]+N,3]-(lx*dL)*round((config[bsloc[i],3]-config[bsloc[i]+N,3])/(lx*dL),0))**2+(config[bsloc[i],4]-config[bsloc[i]+N,4]-(lx*dL)*round((config[bsloc[i],4]-config[bsloc[i]+N,4])/(lx*dL),0))**2)**0.5)
        distances.append(((config[bsloc[i],3]-config[bsloc[i]+N,3]-(lx*dL)*round((config[bsloc[i],3]-config[bsloc[i]+N,3])/(lx*dL),0))**2+(config[bsloc[i],4]-config[bsloc[i]+N,4]-(lx*dL)*round((config[bsloc[i],4]-config[bsloc[i]+N,4])/(lx*dL),0))**2)**0.5)
        num_lst.append([config[bsloc[i],2],config[bsloc[i]+N,2]])
        distances.append(((config[bsloc[i],3]-config[N+N-1,3]-(lx*dL)*round((config[bsloc[i],3]-config[N+N-1,3])/(lx*dL),0))**2+(config[bsloc[i],4]-config[N+N-1,4]-(lx*dL)*round((config[bsloc[i],4]-config[N+N-1,4])/(lx*dL),0))**2)**0.5)
        num_lst.append([config[bsloc[i],2],config[N+N-1,2]])        
        distances.append(((config[bsloc[i],3]-config[N-1,3]-(lx*dL)*round((config[bsloc[i],3]-config[N-1,3])/(lx*dL),0))**2+(config[bsloc[i],4]-config[N-1,4]-(lx*dL)*round((config[bsloc[i],4]-config[N-1,4])/(lx*dL),0))**2)**0.5)
        num_lst.append([config[bsloc[i],2],config[N-1,2]])     
        distances.append(((config[bsloc[i]+N,3]-config[N+N-1,3]-(lx*dL)*round((config[bsloc[i]+N,3]-config[N+N-1,3])/(lx*dL),0))**2+(config[bsloc[i]+N,4]-config[N+N-1,4]-(lx*dL)*round((config[bsloc[i]+N,4]-config[N+N-1,4])/(lx*dL),0))**2)**0.5)
        num_lst.append([config[bsloc[i]+N,2],config[N+N-1,2]])        
        distances.append(((config[bsloc[i]+N,3]-config[N-1,3]-(lx*dL)*round((config[bsloc[i]+N,3]-config[N-1,3])/(lx*dL),0))**2+(config[bsloc[i]+N,4]-config[N-1,4]-(lx*dL)*round((config[bsloc[i]+N,4]-config[N-1,4])/(lx*dL),0))**2)**0.5)
        num_lst.append([config[bsloc[i]+N,2],config[N-1,2]]) 
        
    bonds=[]
    prob=[]
    
    
    for k in range (len(distances)):
        if distances[k]>0 and distances[k]<=2**0.5:
            bonds.append([int(num_lst[k][0]),int(num_lst[k][1])])

    list_combinations = list()
    for n in range(1,len(bsloc)+2):
        list_combinations += list(combinations(bonds, n))
        
    check=[]
    for i in range (len(list_combinations)):
        if len(np.unique(list_combinations[i]))==len(list_combinations[i])*2:
            check.append(0)
        else:
            check.append(1)

    check=np.array(check)
    deletion=np.where(check>0)

    if len(deletion[0])>0:
        list_combinations=np.delete(list_combinations,deletion[0])

    lenth=len(list_combinations)
    list2=[]
    for i in range (lenth):
        list1=np.zeros(lenth)
        for j in range (lenth):
            if i!=j and all(item in list_combinations[j] for item in list_combinations[i]):                   
                list1[j]=1

        list2.append(np.sum(list1))
    list2=np.array(list2)
    deletion=np.where(list2>0)

    if len(deletion[0])>0:
        list_combinations=np.delete(list_combinations,deletion[0])
        
    return (list_combinations,(op/len(bsloc))**0.5,min(bonding))

#compute active interaction matrix#
def active_interactions(chi,ustates,bsloc):
    actchi=np.zeros((5+len(bsloc)*2,5+len(bsloc)*2))
    
    bb1=0 #backbone beads
    sg1=1 #side group

    bb2=2 #backbone beads
    sg2=3 #side group
    
    sol=4 #solvent
    
    actchi[:,sol]=chi[:,sol]
    actchi[sol,:]=chi[sol,:]
    
    actchi[sol,sol]=chi[sol,sol]
    prob=[]

    if len(ustates)>0:
        for i in range (len(ustates)):
            prob1=[]
            for j in range (len(ustates[i])):
                prob1.append(-chi[ustates[i][j][0]-1,ustates[i][j][1]-1])
            
            prob.append(math.exp(sum(prob1)))
        
        
            sel_ustate=random.choices([i for i in range (len(prob))],weights=prob, k=1)[0]
            sel_ustate=ustates[sel_ustate]

        for i in range (len(sel_ustate)):
            actchi[sel_ustate[i][0]-1,sel_ustate[i][1]-1]=chi[sel_ustate[i][0]-1,sel_ustate[i][1]-1]
            actchi[sel_ustate[i][1]-1,sel_ustate[i][0]-1]=chi[sel_ustate[i][1]-1,sel_ustate[i][0]-1]
    return(actchi)


def distance_calc (config,id1,id2,lx,ly,dL):
    return(((config[id1,3]-config[id2,3]-(lx*dL)*round((config[id1,3]-config[id2,3])/(lx*dL),0))**2+(config[id1,4]-config[id2,4]-\
    (lx*dL)*round((config[id1,4]-config[id2,4])/(lx*dL),0))**2)**0.5)


#simulation engine#
def engine (steps, N,M,nsol, config, dL, lx, ly, rho, chi,over,run,configurations,moves_acceptance,atoms,bsloc,seq,bssg,indexes,pbias):
    bb1=0 #backbone beads
    sg1=1 #side group

    bb2=2 #backbone beads
    sg2=3 #side group
    
    sol1=4 #solvent

    energy=[]
    energy1=[]
    results=open('results_N%d_run%d_sol%.1f_bssg%.1f_bsloc%s.txt'%(N,run,chi[bb1,sol1]/2,bssg,seq),'w')
    results.write('step bond_e nb_e bsbs_dist ree1 ree2\n')
    counter=0
    for i in range (steps):  
        counter=counter+1
        if counter==10000:
            
            print_config(config,atoms,configurations,counter)
            counter=0      
        
        if i==0:            
                        
            ree=e2e(config, N, M, lx, ly,dL)
            connections=connectivity (config,bsloc,N,lx,ly,dL)
            ustates=connections[0]
            op=connections[1]
            bonding_dist=connections[2]
            chiact=active_interactions(chi,ustates,bsloc)

            Hbond=bondHamiltonian (config, lx, ly, dL, N, M)
            nonbonded=nbHamiltonian (config,chiact,over,N,lx,ly)        
            Hnb=nonbonded

            
        results.write('%d %f %f %f %f %f\n'%(i, Hbond, Hnb,op,ree[0],ree[1]))  

        propconfig1=config.copy()
        
        # movetype=random.choices([0,1,2,3,4],weights=(0.1,0.4,0.1,0.2,0.4), k=1)
        movetype=random.choices([0,1,2,4],weights=(0.2,0.4,0.1,0.3), k=1)

        if movetype[0]==0:
            propconfig=rept_move (propconfig1, N, lx, ly, dL)
        
        elif movetype[0]==1:
            propconfig=displace_move (propconfig1, lx, ly, N, nsol, dL)
            
        elif movetype[0]==2:            
            propconfig=trans_move (propconfig1, N, nsol, lx, ly, dL)
            
        elif movetype[0]==3:     
            propconfig=swap (propconfig1,N,lx,ly,dL)

        elif movetype[0]==4: 
            move1=assoc_dissoc (propconfig1,N,lx,ly,dL,indexes,bsloc,pbias)
            propconfig=move1[0]
            dist_prop=distance_calc (propconfig,move1[1],move1[2],lx,ly,dL)
            dist=distance_calc (propconfig1,move1[1],move1[2],lx,ly,dL)
            
        connections=connectivity (propconfig,bsloc,N,lx,ly,dL)
        ustates_prop=connections[0]
        op_prop=connections[1]      
        bonding_dist_prop=connections[2]        
        chiact=active_interactions(chi,ustates_prop,bsloc)        
        ree_prop=e2e(propconfig, N, M, lx, ly,dL)        
                               
        Hbond1=bondHamiltonian (propconfig, lx, ly, dL, N,M)
        nonbonded=nbHamiltonian (propconfig,chiact,over,N,lx,ly)
        Hnb1=nonbonded
        
        ran=random.uniform(0,1)
        
        if movetype[0]==4:
            if dist_prop<=2**0.5 and dist>2**0.5:
                
                factor=(1-pbias)*8/pbias/(900-9)
            elif dist_prop>2**0.5 and dist<=2**0.5:
                factor=pbias*(900-9)/(1-pbias)/8
            else:
                factor=1
        else:
            factor=1
            
        if ran<min(1,factor*np.exp(-((Hnb1+Hbond1)-(Hnb+Hbond)))):
            ree=ree_prop          
            config=propconfig.copy()
            Hnb=Hnb1
            Hbond=Hbond1
            op=op_prop
            bonding_dist=bonding_dist_prop
            moves_acceptance.write('%d %d %d\n'%(i, movetype[0], 1))        
        else:
            moves_acceptance.write('%d %d %d\n'%(i, movetype[0], 0))
            
    fc=open('fc_N%d_run%d_sol%.1f_bssg%.1f_bsloc%s.txt'%(N,run,chi[bb1,sol1]/2,chi[bs1,sg1],bsloc),'w')
    for i in range (len(config)):
        fc.write('%d %d %d %f %f\n'%(config[i,0],config[i,1],config[i,2],config[i,3],config[i,4]))                   


     
run=1   
M=2     #number of chains
lx=30   #size in x dimension
ly=30   #size in y dimension

steps= 10000000     #number of MC steps

indexes=[]
for i in range (lx):
    for j in range (ly):
        indexes.append([i,j])
indexes=np.array(indexes)

N=int(sys.argv[1])      #chain length
bssg=float(sys.argv[3])     #bssg interaction strength

###reading the sequence of binding sites from the run script###
bsloc = sys.argv[4][1:len(sys.argv[4])-1]       

bsloc = bsloc.split(',')

bsloc = [int(i) for i in bsloc]     #sequence

###setting up simulation parameters###
rho=0.7         #density
dL=1.0          #lattice site size
pbias=0.5       #association-dissociation bias factor
A=lx*ly
nsol=int(rho*A*dL*dL-2*N)       #number of solvent molecules

atoms=nsol+2*N
config=initial(lx,ly,N,rho,dL,A,nsol,bsloc) #initial configuration

chi=int_matrix (bbsol=float(sys.argv[2])*2, bsloc=bsloc,bssg=bssg)      #interaction matrix
test=1
over=overlap_matrix(test)
seq=[0 for i in range (N)]
for i in range (len(bsloc)):
    seq[bsloc[i]]=1

seq=[str(i) for i in seq]

seq=str("".join(seq))

configurations=open('configurations_N%d_sol%.1f_bssg%.1f_bsloc%s.lammpstrj'%(N,float(sys.argv[2]),bssg,seq),'w')        #"trajectory" dump in lammps format 
moves_acceptance=open('moves_N%d_sol%.1f_bssg%.1f_bsloc%s.txt'%(N,float(sys.argv[2]),bssg,seq),'w') 

sim=engine(steps, N,M,nsol, config, dL, lx, ly, rho, chi,over, run,configurations,moves_acceptance,atoms,bsloc,seq,bssg,indexes,pbias)

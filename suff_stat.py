import numpy as np
import sys
import scipy.io as sio
from scipy.linalg import svd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import pinv
from numpy.random import rand
from collections import defaultdict
#A = lil_matrix((1000,1000))
#A[0, :100] = rand(100)
#A[1, 100:200]=A[0, :100]
#A.setdiag(rand(1000))
#A = A.tocsr()
#(U,s,V)=svds(A,50)
#print s
#sio.savemat('dump.mat',dict(A=A))

def makevocab(infile):
    ctr=1
    vocab = defaultdict(int)
    infl = open(infile)
    for line in infl:
        for a in line.strip().split():
            if not(a in vocab):
                vocab[a] = ctr    
                ctr += 1
    infl.close()
    return vocab

def get_mat_counts(vocab,infile,formal=False):
    P1 = np.zeros(len(vocab))
    P_21 = lil_matrix((len(vocab),len(vocab)))
    P_31 = [lil_matrix((len(vocab),len(vocab))) for k in range(len(vocab))]
    infl = open(infile)
    for line in infl:
        a= line.strip().split()
        seql=len(a)
        if(formal):
            seql=3
        for i in range(seql):
            P1[vocab[a[i]]-1]+= 1
            if not(i>(len(a)-2)):
                #print i,len(vocab),line[i],line[i+1]
                P_21[vocab[a[i]]-1,vocab[a[i+1]]-1]+=1
            if not(i>(len(a)-3)):
                P_31[vocab[a[i+1]]-1][vocab[a[i+2]]-1,vocab[a[i]]-1] += 1
    infl.close()
    return (P1,P_21,P_31)

def infer(b1,binf,B,seq,vocab):
    prod = binf.transpose()
    for i in range(len(seq)):
        prod = prod.dot(B[vocab[seq[len(seq)-1-i]]-1])
    prod = prod.dot(b1)
    return prod

def marginal(b1,binf,B,seq,vocab):
    return infer(b1,binf,B,seq,vocab)

def conditional(b1,binf,B,seq,vocab,word):
    den = infer(b1,binf,B,seq,vocab)
    prod = b1
    for i in range(len(seq)):
        prod = (B[vocab[seq[i]]-1]).dot(prod)
    prod= prod/(den*1.0)
    prod = ((binf.transpose()).dot(B[vocab[word]-1])).dot(prod)
    return prod

def get_params(U, P_3x1, P_31, word):
    aux = (U.transpose().dot(P_3x1[vocab[word]-1].toarray())).dot(pinv(U.transpose().dot(P_31.toarray()))) 
    U1,s1,V1 = svd(aux)
    return s1
if __name__=="__main__":
    infile = sys.argv[1]
    dimm = int(sys.argv[2])
    vocab = makevocab(infile)
    P_31 = lil_matrix((len(vocab),len(vocab)))
    P_3x1 = [lil_matrix((len(vocab),len(vocab))) for k in range(len(vocab))]
    p1,p_21,p_31 = get_mat_counts(vocab,infile)
    P1 = p1/(p1.sum()*1.0)
    P_21 = p_21/(p_21.sum()*1.0)
    for k in range(len(vocab)):
        P_3x1[k]=p_31[k]*P1[k]/(p_31[k].sum()*1.0)
        P_31 = P_31 + p_31[k]
    P_31 = P_31/(P_31.sum()*1.0)
    U,s,V=svds(P_21,dimm)
    b1 = U.transpose().dot(P1)
    binf = (pinv(P_21.transpose().dot(U))).dot(P1)
    B=[]
    for k in range(len(vocab)):
        Bx = (U.transpose().dot(P_3x1[k].toarray())).dot(pinv(U.transpose().dot(P_21.toarray())))
        B.append(Bx)
    #print b1, binf
    #print B
    te_file = open(sys.argv[3])
    
    # get marginal probability
    #for line in te_file:
    #    seq = line.strip().split()
    #    print marginal(b1,binf,B,seq,vocab)
    
    # get conditional probabilities
    #seq = ['2','2','5']
    #print conditional(b1,binf,B,seq,vocab,'1')
    #print conditional(b1,binf,B,seq,vocab,'2')
    #print conditional(b1,binf,B,seq,vocab,'3')
    #print conditional(b1,binf,B,seq,vocab,'4')
    #print conditional(b1,binf,B,seq,vocab,'5')

    # get operators
    print get_params(U,P_3x1,P_31,'1')
    print get_params(U,P_3x1,P_31,'2')
    print get_params(U,P_3x1,P_31,'3')
    print get_params(U,P_3x1,P_31,'4')
    print get_params(U,P_3x1,P_31,'5')


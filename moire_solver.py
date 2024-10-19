import numpy as np
import matplotlib.pyplot as plt
# from read_DFT_data import *
import os
import gzip
import shutil
import h5py

from Wannier_aux import *


def generate_kpath(HSP_list, Nk=20):
    # for given kpt_list=[k0, k1, ..., k0], generate kpoints on the path.
    # each path has Nk+1 pts, from ki to ki+1
    kpath = []
    for k1, k2 in zip(HSP_list[0:-1], HSP_list[1:]):
        kstep = (k2 - k1) / Nk
        for ik in range(Nk + 1):
            kpt = k1 + ik * kstep
            kpath.append(kpt)
    return np.asarray( kpath )

def gen_lenk(klist):
    # for given kpt_list=[k0, k1, ..., k0], generate kpoints on the path.
    # each path has Nk+1 pts, from ki to ki+1

    len_it = 0
    lenk = []
    for i in range(0, len(klist) - 1):
        lenk.append(len_it)
        len_it += np.linalg.norm(klist[i + 1] - klist[i])
    lenk.append(len_it)

    return np.asarray(lenk)


class Mtwist:
    def __init__(self,para, q0, theta= 3.89, which_model = 'AA'):
        # q0 in the unit of A^{-1}
        self.theta = theta
        self.theta_rad =  theta *2.0*np.pi/360.0
        self.model = which_model

        self.q0 = np.asarray([1.0, 0.0])
        self.q1 = np.asarray([-0.5, 0.5 * np.sqrt(3.0)])
        self.q2 = np.asarray([-0.5, -0.5 * np.sqrt(3.0)])

        self.b1 = np.asarray([2.0, 0.0])
        self.b2 = np.asarray([1.0, np.sqrt(3.0)])
        self.bmat = np.asarray([self.b1, self.b2])

        if( which_model == 'AA'):
            ww1 = para['ww1']
            ww2 = para['ww2']
            ww3 = para['ww3']
            ww4 = para['ww4']
            ww5 = para['ww5']
            ww6 = para['ww6']
            ww7 = para['ww7']
            ww8 = para['ww8']
            ww9 = para['ww9']
            ww10 = para['ww10']

            w1 = para['w1']
            w2 = para['w2']
            w3 = para['w3']
            w4 = para['w4']
            w5 = para['w5']
            w6 = para['w6']

            self.mx = para['mx']
            self.my = para['my']

            self.moire_pot = []

            dq = -self.b1
            lind = [-1,-1]
            spinind = 0
            self.moire_pot.append(  [dq, lind, spinind, ww5 + 1j * ww2 ])

            dq = -self.b1
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ww5 + 1j * ww2])

            dq = -self.b1 + self.b2
            lind = [-1, -1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, 1j*ww1 + ww4 - 1j * ww7 - ww10])

            dq = -self.b1 + self.b2
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, 1j * ww1 + ww4 + 1j * ww7 + ww10])

            dq = -self.b2
            lind = [-1, -1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, 1j * ww1 + ww4 + 1j * ww7 + ww10])

            dq = -self.b2
            lind = [-1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, w2 - 1j * w1])
            spinind = 2
            self.moire_pot.append([dq, lind, spinind, w4 + 0.5 * (np.sqrt(3.0) + 1j) * w6 ] )
            spinind = 3
            self.moire_pot.append([dq, lind, spinind, w5 -1j * w3 ])

            dq = -self.b2
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, 1j*ww1 + ww4 - 1j*ww7 - ww10 ])

            dq = 0.0
            lind = [-1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind,  ww3])
            spinind = 1
            self.moire_pot.append([dq, lind, spinind, 0.5*(-ww8-ww9)])
            spinind = 2
            self.moire_pot.append([dq, lind, spinind, 1j*0.5*(ww8-ww9)])
            spinind = 3
            self.moire_pot.append([dq, lind, spinind, -1j*ww6])

            dq = self.b2
            lind = [-1, -1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind,  -1j*ww1 + ww4 - 1j*ww7 + ww10])

            dq = self.b2
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, -1j * ww1 + ww4 + 1j * ww7 - ww10])

            dq = self.b1 - 2 * self.b2
            lind = [-1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ww3])
            spinind = 1
            self.moire_pot.append([dq, lind, spinind, 0.5*(ww8 + ww9)])
            spinind = 2
            self.moire_pot.append([dq, lind, spinind, 0.5 * 1j * (ww8 - ww9)])
            spinind = 3
            self.moire_pot.append([dq, lind, spinind, - 1j * (ww6)])

            dq = self.b1 -  self.b2
            lind = [-1, -1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, -1j*ww1 + ww4 + 1j*ww7 -ww10])

            dq = self.b1 - self.b2
            lind = [-1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind,  w2 + 1j * w1])
            spinind = 2
            self.moire_pot.append([dq, lind, spinind, -w4 - 0.5 * (np.sqrt(3.0)-1j)* w6])
            spinind = 3
            self.moire_pot.append([dq, lind, spinind, -w5 - 1j * w3 ])

            dq = self.b1 - self.b2
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, -1j*ww1 + ww4 - 1j * ww7 + ww10])

            dq = self.b1
            lind = [-1, -1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ww5 - 1j *ww2])

            dq = self.b1
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ww5 - 1j * ww2])








        else:
            self.moire_pot = []

            ww1 = para['ww1']
            ww2 = para['ww2']
            ww3 = para['ww3']
            ww4 = para['ww4']
            ww5 = para['ww5']
            ww6 = para['ww6']
            ww7 = para['ww7']
            ww8 = para['ww8']
            ww9 = para['ww9']
            ww10 = para['ww10']
            ww11 = para['ww11']
            ww12 = para['ww12']

            w1 = para['w1']
            w2 = para['w2']
            w3 = para['w3']
            w4 = para['w4']

            self.mx = para['mx']
            self.my = para['my']

            dq = -self.b1
            lind = [-1,-1]
            spinind = 0
            self.moire_pot.append(  [dq, lind, spinind, ww7 - 1j * ww3 ])


            dq = -self.b1
            lind = [1, 1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ww7 + 1j * ww3])

            dq = -self.b1 + self.b2
            lind = [-1,-1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, -1j*ww2 + ww6 + 1j * ww10 - ww12 ])

            dq = -self.b1 + self.b2
            lind = [1, 1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, 1j * ww2 + ww6 + 1j * ww10 + ww12])

            dq = -self.b2
            lind = [-1, -1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind,- 1j * ww2 + ww6 - 1j * ww10 + ww12])

            dq = -self.b2
            lind = [-1, +1]

            spinind = 0
            self.moire_pot.append([dq, lind, spinind,  w2] )
            spinind = 1
            self.moire_pot.append([dq, lind, spinind,  -1j*w1] )
            spinind = 2
            self.moire_pot.append([dq, lind, spinind, w3])
            spinind = 3
            self.moire_pot.append([dq, lind, spinind, -1j*w4])

            dq = -self.b2
            lind = [+1,+1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, 1j*ww2 + ww6 - 1j * ww10 - ww12])

            dq = -0.0
            lind = [-1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, (ww4 +  1j * ( ww9))])
            spinind = 1
            self.moire_pot.append([dq, lind, spinind, -ww11  - 1j * ww1 ])
            spinind = 3
            self.moire_pot.append([dq, lind, spinind,  ( ww5 - 1j * (ww8))])


            dq = self.b2
            lind = [-1,-1]
            spinind = 0
            self.moire_pot.append( [dq, lind, spinind, (1j * ww2 + ww6 + 1j * ww10 + ww12)])

            dq = self.b2
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, (-1j * ww2 + ww6 + 1j * ww10 - ww12)])

            dq = self.b1 - 2 * self.b2
            lind = [-1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ( ww4 - 1j*ww9)])
            spinind = 1
            self.moire_pot.append([dq, lind, spinind, ww11 - 1j * ww1 ])
            spinind = 3
            self.moire_pot.append([dq, lind, spinind, 0.5 * (-ww5 - 1j * (ww8))])


            dq = self.b1 - self.b2
            lind = [-1,-1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, 1j * ww2 + ww6  - 1j * ww10 - ww12] )

            dq = self.b1 - self.b2
            lind = [-1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, w2])
            spinind = 1
            self.moire_pot.append([dq, lind, spinind, -1j*w1])
            spinind = 2
            self.moire_pot.append([dq, lind, spinind, - w3])
            spinind = 3
            self.moire_pot.append([dq, lind, spinind, -1j * w4])

            dq = self.b1 - self.b2
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, -1j * ww2 + ww6 - 1j * ww10 + ww12])

            dq = self.b1
            lind = [-1,-1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ww7 + 1j * ww3])

            dq = self.b1
            lind = [+1, +1]
            spinind = 0
            self.moire_pot.append([dq, lind, spinind, ww7 - 1j * ww3])

        moire_pot_half = [] # I then need to remove the componten that is hermitian of one other
        for i in range(0,len(self.moire_pot)):
            dq, lind, spinind, w = self.moire_pot[i]
            if_rep = False
            for j in range(0,len(moire_pot_half)):
                dq2, lind2, spinind2, w2 = moire_pot_half[j]

                if( np.linalg.norm(dq + dq2 ) > 0.0001 ): continue
                if( lind[0] != lind2[1] or lind[1] != lind2[0] ): continue
                if( spinind != spinind2 ): continue
                if_rep = True
                # print('REP')
                #
                # print(dq, np.round(dq@np.linalg.inv(self.bmat)), lind, spinind, w)
                # print(dq2, np.round(dq@np.linalg.inv(self.bmat)), lind2, spinind2, w2)
                break
            if( not if_rep):
                moire_pot_half.append( [dq, lind, spinind, w ])
        self.moire_pot = moire_pot_half
        self.if_su2 = para['if_su2']
        self.if_zero_twist_kinetic = para['if_zero_twist_kinetic']







            # take q0=1 unit
        self.mx = self.mx/ ( q0**2 )
        self.my = self.my/ ( q0** 2 )



        b3d = np.zeros((3,3))
        b3d[0:2,0:2] = self.bmat
        b3d[2,2] =  2.0 *np.pi
        avec = avec_to_bvec(b3d)
        self.amat = avec[0:2,0:2]

        self.a0 = np.linalg.norm(self.amat[0]) *  ( 2.0 * np.sin( self.theta_rad/2.0) ) # non-twist lattice constant

        self.Gam = np.asarray([0.0, 0.0])
        #         self.M = np.asarray([0.5,0.5*np.sqrt(3.0)])
        self.M = np.asarray([1.0, 0.0])
        self.K = 2.0 / np.sqrt(3.0) * np.asarray([0.5 * np.sqrt(3.0), 0.5])

        self.HSPs = np.asarray([self.Gam, self.M, self.K, self.Gam])
        self.lenk_label_pos = []
        len_ = 0.0
        for i in range(0, len(self.HSPs) - 1):
            self.lenk_label_pos.append(len_)
            dk = (self.HSPs[i + 1] - self.HSPs[i])
            len_ += np.linalg.norm(dk)
        self.lenk_label_pos.append(len_)


        self.HSP_label = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']
        self.nkline = 20
        self.kpath = generate_kpath(self.HSPs, Nk=self.nkline)
        self.lenk = gen_lenk(self.kpath)

        self.spin_pauli = np.zeros((4,2,2), dtype=np.complex128)
        self.spin_pauli[0] = ( np.asarray([[1,0],[0,1]]))
        self.spin_pauli[1] = (np.asarray([[0, 1], [1, 0]]))
        self.spin_pauli[2] = (np.asarray([[0, -1j], [1j, 0]]))
        self.spin_pauli[3] = (np.asarray([[1, 0], [0, -1]]))


        self.get_Qlat(21, 8)
        self.get_h_moire()

    #
    # def update_para(self,w12, phase, ww3):
    #     W = w12
    #     psi = phase
    #
    #     self.w1 = W * np.sin(psi)
    #     self.w2 = W * np.cos(psi)
    #     self.ww3 = ww3  #
    #
    #     self.get_h_moire()
    #

    # def get_kbz(self, Nz):
    def import_Qlat(self, Q):
        self.Q_tot = np.copy(Q)
        self.Q_sign = []
        for iq, Q in enumerate(self.Q_tot):
            Q_b12 = Q @ np.linalg.inv(self.bmat)
            Qind = Q_b12 - np.asarray([-0.5, 0.5])
            if (np.linalg.norm(Qind - np.round(Qind).astype(np.int32)) < 0.05):
                Qind = np.round(Qind).astype(np.int32)
                self.Q_sign.append(+1)
                continue
            Qind = Q_b12 - np.asarray([0.0, 0.5])
            if (np.linalg.norm(Qind - np.round(Qind).astype(np.int32)) < 0.05):
                Qind = np.round(Qind).astype(np.int32)
                self.Q_sign.append(-1)
                continue
            Qind = Q_b12 - np.asarray([-0.5, 0.5])
            print(Qind, np.round(Qind).astype(np.int32))
            print(Q_b12 - np.asarray([0.0, 0.5]))

            raise ValueError('Cannot find the Q site', Q_b12)

        self.Q_sign = np.asarray(self.Q_sign)

        self.get_h_moire()

    def generate_label(self, ax):

        klables = self.HSP_label
        nkline = self.nkline
        for i in range(0, len(klables)):
            # ax.axvline(self.lenk[i * nkline], linestyle='--', color='black')
            ax.axvline(self.lenk_label_pos[i], linestyle='--', color='black')

        # ax.set_xticks([self.lenk[i * nkline] for i in range(0, len(klables))], klables, font=labelfont)
        ax.set_xticks(self.lenk_label_pos, klables)

        ax.set_ylabel('$E$ (meV)')

    def update_Qlat(self, Nq, cutoff):
        self.get_Qlat(Nq, cutoff)
        self.get_h_moire()

    def get_Qlat(self, Nq, cutoff):

        Qtot = []
        Qsign = []
        for i in range(-Nq, Nq + 1):
            for j in range(-Nq, Nq + 1):
                q = (i - 0.5) * self.b1 + (j + 0.5) * self.b2
                if (np.linalg.norm(q) < cutoff):
                    Qtot.append(q)
                    Qsign.append(+1)

                q = i * self.b1 + (j - 0.5) * self.b2
                if (np.linalg.norm(q) < cutoff):
                    Qtot.append(q)
                    Qsign.append(-1)

        self.Q_tot = np.asarray(Qtot)
        self.Q_sign = np.asarray(Qsign)

    def get_h_moire(self):

        # if( self.model == 'AA'):
        #     Qcut = 0.001
        #     self.h1 = np.zeros((len(self.Q_tot), len(self.Q_tot)), dtype=np.complex128)
        #     for i in range(0, len(self.Q_tot)):
        #         for j in range(0, len(self.Q_tot)):
        #             Q1 = self.Q_tot[i]
        #             Q2 = self.Q_tot[j]
        #
        #             if (np.linalg.norm(Q1 + self.q0 - Q2) < Qcut):
        #                 self.h1[i, j] += 1j * self.w1 + self.w2
        #
        #             if (np.linalg.norm(Q1 - self.q0 - Q2) < Qcut):
        #                 self.h1[i, j] += -1j * self.w1 + self.w2
        #
        #             if (np.linalg.norm(Q1 + self.q2 - self.q1 - Q2) < Qcut):
        #                 self.h1[i, j] += self.ww3
        #
        #             if (np.linalg.norm(Q1 - self.q2 + self.q1 - Q2) < Qcut):
        #                 self.h1[i, j] += self.ww3
        #
        #             dq = self.q0 - self.q1 + self.q2
        #             if (np.linalg.norm(Q1 + dq - Q2) < Qcut):
        #                 self.h1[i, j] += -1j * self.ww1 + self.ww4
        #
        #             if (np.linalg.norm(Q1 - dq - Q2) < Qcut):
        #                 self.h1[i, j] += 1j * self.ww1 + self.ww4
        #
        #             dq = + self.q0 + self.q1 - self.q2
        #             if (np.linalg.norm(Q1 + dq - Q2) < Qcut):
        #                 self.h1[i, j] += -1j * self.ww1 + self.ww4
        #
        #             if (np.linalg.norm(Q1 - dq - Q2) < Qcut):
        #                 self.h1[i, j] += 1j * self.ww1 + self.ww4
        #
        #             dq = 2.0 * self.q0
        #             if (np.linalg.norm(Q1 + dq - Q2) < Qcut):
        #                 self.h1[i, j] += -1j * self.ww2 + self.ww5
        #
        #             if (np.linalg.norm(Q1 - dq - Q2) < Qcut):
        #                 self.h1[i, j] += + 1j * self.ww2 + self.ww5
        #
        #
        #
        #             dq = self.bmat[1] - self.bmat[0]
        #             if (np.linalg.norm(Q1 + dq - Q2) < Qcut):
        #                 if( self.Q_sign[i] != self.Q_sign[j] ): raise ValueError('Invalid')
        #                 self.h1[i, j] +=  self.ww10 * self.Q_sign[i]
        #
        #             dq = -self.bmat[1] + self.bmat[0]
        #             if (np.linalg.norm(Q1 + dq - Q2) < Qcut):
        #                 if (self.Q_sign[i] != self.Q_sign[j]): raise ValueError('Invalid')
        #                 self.h1[i, j] += self.ww10 * self.Q_sign[i]
        #
        #             dq = -self.bmat[1]
        #             if (np.linalg.norm(Q1 + dq - Q2) < Qcut):
        #                 if (self.Q_sign[i] != self.Q_sign[j]): raise ValueError('Invalid')
        #                 self.h1[i, j] +=- self.ww10 * self.Q_sign[i]
        #
        #             dq =  self.bmat[1]
        #             if (np.linalg.norm(Q1 + dq - Q2) < Qcut):
        #                 if (self.Q_sign[i] != self.Q_sign[j]): raise ValueError('Invalid')
        #                 self.h1[i, j] +=- self.ww10 * self.Q_sign[i]
        #
        #
        #
        #
        #     self.h1 = np.reshape(self.h1, (1, len(self.Q_tot), len(self.Q_tot)))
        # else:

        Qcut = 0.001
        self.h1 = np.zeros((2*len(self.Q_tot), 2*len(self.Q_tot)), dtype=np.complex128)
        for i in range(0, len(self.Q_tot)):
            for j in range(0, len(self.Q_tot)):
                Q1 = self.Q_tot[i]
                Q2 = self.Q_tot[j]
                qshift = [self.q2, self.q1]
                for dq, lind, spinindx, wcoup in self.moire_pot:
                    l1 , l2 = lind[0], lind[1]
                    qsign1, qsign2 = self.Q_sign[i], self.Q_sign[j]
                    if( not ( np.abs(qsign1 - l1 ) < 0.01 and np.abs(qsign2-l2)<0.01 ) ): continue

                    if (l1 < 0): q1 = qshift[0]
                    else: q1 = qshift[1]
                    if (l2 < 0): q2 = qshift[0]
                    else: q2 = qshift[1]

                    dq_shift = dq + q2 - q1
                    if( np.linalg.norm(dq_shift - (Q2-Q1) ) > Qcut ): continue
                    spin = self.spin_pauli[spinindx]
                    self.h1[2*i:2*i+2, 2*j: 2*j+2] += spin * wcoup

                    #
                    # dQ = Q2-Q1
                    # if( np.linalg.norm(dQ- np.asarray([-1,0])) < Qcut or np.linalg.norm(dQ- np.asarray([1,0])) < Qcut ):
                    #     if( self.Q_sign[i] <0 and self.Q_sign[j] > 0 ):
                    #         self.h1[ 2*i: 2*i+2, 2*j:2*j+2] += self.spin_pauli[0] * self.para['w2']
                    #
                    # if (np.linalg.norm(dQ - np.asarray([0, np.sqrt(3.0)])) < Qcut ):
                    #     if (self.Q_sign[i] < 0 and self.Q_sign[j] > 0):
                    #         self.h1[2 * i: 2 * i + 2, 2 * j:2 * j + 2] += 0.5*(self.spin_pauli[0]+self.spin_pauli[3]) * self.para['ww4']
                    #
                    # if (np.linalg.norm(dQ - np.asarray([0, -np.sqrt(3.0)])) < Qcut):
                    #     if (self.Q_sign[i] < 0 and self.Q_sign[j] > 0):
                    #         self.h1[2 * i: 2 * i + 2, 2 * j:2 * j + 2] += 0.5 * (
                    #                     self.spin_pauli[0] - self.spin_pauli[3]) * self.para['ww4']

            # div = np.max( np.abs(self.h1 - np.conj(np.transpose(self.h1))))
            # print('div from her', div)
        self.h1 = self.h1 + np.conj( np.transpose(self.h1))

        if( self.if_su2): # kill one of spin
            h1 = np.reshape(self.h1, ( len(self.Q_tot),2,len(self.Q_tot),2) )
            self.h1 = h1[:,0, :,0]
            if( np.max(np.abs(h1[:,0,:,1]))>1e-3):
                print( np.max(np.abs(h1[:,0,:,1])) )
                raise ValueError('Donnot have su2')
            self.h1 = np.reshape(self.h1, (1, len(self.Q_tot),  len(self.Q_tot)))

        else:
            self.h1 = np.reshape(self.h1, (1, 2*len(self.Q_tot), 2*len(self.Q_tot)))


    def get_h0(self, k):


        if( self.if_zero_twist_kinetic):
            Rp = np.asarray([[1.,0.],[0.,1.]], dtype=np.complex128)
            Rn = np.asarray([[1., 0.], [0., 1.]], dtype=np.complex128)
        else:
            Rp = np.asarray([[np.cos(self.theta_rad/2.0 ), -np.sin(self.theta_rad/2.0 )],
                            [np.sin(self.theta_rad/2.0 ), np.cos(self.theta_rad/2.0 )]], dtype=np.complex128)
            Rn = np.asarray([[np.cos(self.theta_rad/2.0 ), np.sin(self.theta_rad/2.0 )],
                        [-np.sin(self.theta_rad/2.0 ), np.cos(self.theta_rad/2.0 )]], dtype=np.complex128)


        if( self.if_su2):
            h0 = np.zeros((len(k), len(self.Q_tot), len(self.Q_tot)), dtype=np.complex128)
            for i in range(0, len(self.Q_tot)):
                if(self.Q_sign[i] > 0.0):
                    kq = ( k[:, :] - self.Q_tot[i, :] ) @ (Rp)
                else:
                    kq = ( k[:, :] - self.Q_tot[i, :] ) @ (Rn)
                h0[:, i, i] += (kq[:, 0]) ** 2 / (2.0 * self.mx) + (kq[:, 1]) ** 2 / (
                        2.0 * self.my)
        else:
            h0 = np.zeros((len(k), 2*len(self.Q_tot), 2*len(self.Q_tot)), dtype=np.complex128)
            for i in range(0, len(self.Q_tot)):
                if (self.Q_sign[i] > 0.0):
                    kq = (k[:, :] - self.Q_tot[i, :]) @ (Rp)
                else:
                    kq = (k[:, :] - self.Q_tot[i, :]) @ (Rn)

                h0[:, 2*i, 2*i] += (kq[:, 0]) ** 2 / (2.0 * self.mx) + (kq[:, 1]) ** 2 / (
                        2.0 * self.my)

                h0[:, 2 * i+1, 2 * i+1] += (kq[:, 0]) ** 2 / (2.0 * self.mx) + (kq[:, 1]) ** 2 / (
                        2.0 * self.my)

        return h0

    def get_hk(self, k, alpha=1.0):

        hk = self.get_h0(k) + alpha * self.h1
        return hk

    def construct_real_space_lattice(self, N=300, cutoff=15):

        a0 = self.a0
        theta = self.theta_rad

        rlist = []
        for i in range(-N, N):
            for j in range(-N, N):
                r = i * a0 * np.asarray([1.0, 0.0]) + j * a0 * np.asarray([-0.5, 0.5 * np.sqrt(3.0)])
                if (np.linalg.norm(r) <= cutoff * a0):
                    rlist.append(r)
        rlist = np.asarray(rlist)

        #         theta = 0.0
        rot = np.asarray([[np.cos(theta / 2.0), - np.sin(theta / 2.0)],
                          [np.sin(theta / 2.0), np.cos(theta / 2.0)]])

        rtop = rlist @ np.transpose(rot)  # top rotated by -theta/2
        rbot = rlist @ (rot)

        self.rbot = rbot
        self.rtop = rtop

    def get_charge_distribution(self, kbz, nband=1, N=300, cutoff=15):
        self.construct_real_space_lattice(N, cutoff)

        hk = self.get_hk(kbz)
        eg, egv = np.linalg.eigh(hk)

        rbot, rtop = self.rbot, self.rtop

        rho_bot_arr = []
        rho_top_arr = []
        # fig, axs_all = plt.subplots(ncols=2, nrows=3, figsize=(10, 14))
        for iband in range(0, nband):

            vf = egv[:, :, iband]

            Q_tot = self.Q_tot
            Q_sign = self.Q_sign

            N_moire = 1 / (2.0 * np.sin(self.theta_rad / 2.0)) ** 2  # number of atoms per moire unit cell
            Nbz = len(kbz)
            Nall = Nbz * N_moire

            ind_p = Q_sign > 0.0
            ind_n = Q_sign < 0.0

            Qtop = Q_tot[ind_p]
            Qbot = Q_tot[ind_n]

            QQtop = np.reshape(Qtop, (-1, 1, 2)) - np.reshape(Qtop, (1, -1, 2))
            QQbot = np.reshape(Qbot, (-1, 1, 2)) - np.reshape(Qbot, (1, -1, 2))

            form_bot = np.zeros((len(Qbot), len(Qbot), len(rbot)), dtype=np.complex128)
            form_top = np.zeros((len(Qtop), len(Qtop), len(rtop)), dtype=np.complex128)

            for ir in range(0, len(rbot)):
                form_bot[:, :, ir] = np.exp(1j * np.tensordot(QQbot, rbot[ir], axes=1))

            for ir in range(0, len(rtop)):
                form_top[:, :, ir] = np.exp(1j * np.tensordot(QQtop, rtop[ir], axes=1))

            rho_top = np.einsum('kq, kp, qpr->r', np.conj(vf[:, ind_p]), vf[:, ind_p], form_top) / Nall

            rho_bot = np.einsum('kq, kp, qpr->r', np.conj(vf[:, ind_n]), vf[:, ind_n], form_bot) / Nall

            rho_bot_arr.append(rho_bot)
            rho_top_arr.append(rho_top)

        return rbot, rtop, np.asarray(rho_bot_arr), np.asarray(rho_top_arr)

    def get_berry(self, bandind, dk, N, kbz = None ):

        if( kbz is None ):
            kbz = get_kbz_2d(N) @ self.bmat
            self.kbz = kbz
        else:
            kbz = kbz
            self.kbz = kbz

        get_hk = lambda k: self.get_hk(np.reshape(k, (1, -1)))[0]

        berr_arr = []

        for i in range(0, len(kbz)):
            if (i % 30 == 0): print(i, len(kbz))

            k = kbz[i]
            kx = kbz[i] + np.asarray([dk, 0])
            ky = kbz[i] + np.asarray([0, dk])

            hk = get_hk(k)
            hkx = get_hk(kx)
            hky = get_hk(ky)

            eg, egv = np.linalg.eigh(hk)
            egx, egvx = np.linalg.eigh(hkx)
            egy, egvy = np.linalg.eigh(hky)

            egv = egv[:, bandind]
            egvx = egvx[:, bandind]
            egvy = egvy[:, bandind]

            proj = np.einsum('in, nj->nij', egv, np.conj(np.transpose(egv)))
            projx = np.einsum('in, nj-> nij', egvx, np.conj(np.transpose(egvx)))
            projy = np.einsum('in, nj->nij', egvy, np.conj(np.transpose(egvy)))

            dxproj = (projx - proj) / dk
            dyproj = (projy - proj) / dk

            berry = np.einsum('nij, njm, nmi->n', proj, dxproj, dyproj)
            berry += -np.einsum('nij, njm, nmi->n', proj, dyproj, dxproj)
            berry = berry * 1j

            berr_arr.append(berry)

        berr_arr = np.asarray(berr_arr)

        return kbz, berr_arr



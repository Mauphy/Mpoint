import numpy as np
import matplotlib.pyplot as plt
# from read_DFT_data import *
import os
import gzip
import shutil
import h5py
import pandas as pd
import numba

from collections import defaultdict

from scipy.spatial import KDTree

from Wannier_aux import *
from moire_solver import *
from Smooth_Gauge import *



def get_hop_from_hk(Rlist, hk, klist):
    hop = np.zeros((len(Rlist)), dtype=np.complex128)

    for i, R in enumerate(Rlist):
        kR = klist @ R
        hop[i] = np.sum(np.exp(2.0 * np.pi * (-1j) * kR) * hk) / len(klist)

    return hop


c3z = np.asarray([[np.cos(2.0*np.pi/3.0), -np.sin(2.0*np.pi/3.0)],
                     [np.sin(2.0*np.pi/3.0), np.cos(2.0*np.pi/3.0)]])


def get_vk_all_wal(disp, vk_input, WC_shift):
    kbz = disp.kbz

    vk_allval = [vk_input]

    for g in [c3z, c3z @ c3z]:
        vval = np.zeros_like(vk_input)
        for layer in [-1, 1]:
            indQ = disp.Q_sign * layer > 0.0
            Q = disp.Q_tot[indQ]

            kQ = np.reshape(np.reshape(kbz, (-1, 1, 2)) - np.reshape(Q, (1, -1, 2)), (-1, 2))

            find_mom = KDTree(np.real(kQ))
            mom_new = kQ @ np.transpose(np.linalg.inv(g))
            dis_, ind_ = find_mom.query(np.real(mom_new))
            vk_layer = vk_input[:, indQ]
            vk_layer = np.reshape(vk_layer, -1)
            vnew = vk_layer[ind_]
            ind_cannotfind = dis_ > np.linalg.norm(disp.bmat[0]) / len(kbz)
            vnew[ind_cannotfind] = 0.0
            vnew = np.reshape(vnew, (len(kbz), len(Q)))
            vval[:, indQ] = vnew
        vk_allval.append(vval)
    vk_allval = np.asarray(vk_allval)
    for iv in range(0, len(vk_allval)):
        form = np.exp(- 1j * kbz @ WC_shift[iv])
        vk_allval[iv] = vk_allval[iv] * np.reshape(form, (-1, 1))

    return vk_allval


def get_Mform_factor(kbz, vk1, vk2, q, G, disp):
    # get v_1_kQ^* v_2_(k+q)(Q-G)

    formfact = np.zeros((len(kbz), len(q), len(G)), dtype=np.complex128)

    for sign in [-1, 1]:
        ind_ = disp.Q_sign * sign > 0
        Q = disp.Q_tot[ind_]
        vf1 = vk1[:, ind_]
        vf2 = vk2[:, ind_]

        kQ = np.reshape(kbz, (-1, 1, 2)) - np.reshape(Q, (1, -1, 2))
        kQ = np.reshape(kQ, (-1, 2))
        vf1 = np.reshape(vf1, (-1,))
        vf2 = np.reshape(vf2, (-1,))

        find_mom = KDTree(np.real(kQ))

        for iq, qit in enumerate(q):
            for iG, Git in enumerate(G):
                mom_new = kQ + qit + Git
                dis, ind_ = find_mom.query(np.real(mom_new))

                cutoff = np.linalg.norm(disp.bmat[0]) / len(kbz)
                ind_cannotfind = dis > cutoff

                v2 = np.copy(vf2[ind_])
                v2[ind_cannotfind] = 0.0  # larger than max dis and set to zero

                vv = np.conj(vf1) * v2
                vv = np.reshape(vv, (len(kbz), len(Q)))
                formfact[:, iq, iG] += np.sum(vv, axis=1)
    return formfact



def run_parameter(comp = 'SnSe2', stack = 'AA', folder='WANNIER_PARA/'):
    def get_Vq(q):
        xi = 100  # anst
        Uxi = 24  # meV
        qabs = np.linalg.norm(q, axis=1) * Lat_Para['q0']

        x = xi * qabs
        Vq = 2.0 * np.pi * xi ** 2 * Uxi * np.tanh(x / 2.0) / x
        ind_ = x < 0.0001

        Vq[ind_] = 2.0 * np.pi * xi ** 2 * Uxi * 0.5  # samll x limit
        return Vq

    def get_tilde_V_mom_space(disp, k1, k2, q, G, M1, M2):
        # m1, m2 the form factor for two valleys
        V = np.zeros((len(q), len(k1), len(k2)), dtype=np.complex128)

        qG = np.reshape(np.reshape(q, (-1, 1, 2)) + np.reshape(G, (1, -1, 2)), (-1, 2))
        VqG = np.reshape(get_Vq(qG), (len(q), len(G)))

        for i in range(0, len(k1)):
            for j in range(0, len(k2)):
                mk1 = M1[i]
                mk2 = M2[j]

                mmqG = np.reshape(mk1 * np.conj(mk2), (len(q), len(G)))
                mmqG = np.sum(mmqG * VqG, axis=1) / Lat_Para['Omega0']

                V[:, i, j] = mmqG

        return V
    Lat_Para_dict = {}
    me = 0.000131224  # meV^{-1} * A^{-2}

    # SnSe2, Full model

    theta = 6.01
    mx = me * 0.26
    my = me * 0.77

    w1 = -4.43
    w2 = -78.43
    w3 = -0.50
    w4 = -0.21

    ww1 = -0.67
    ww2 = 9.24
    ww3 = -14.94
    ww4 = 35.88
    ww5 = 0.37
    ww6 = -7.71
    ww7 = -10.59
    ww8 = -0.36
    ww9 = -4.01
    ww10 = 3.94
    ww11 = -0.34
    ww12 = -20.41

    a0 = 3.811
    M = 0.5 * 4.0 * np.pi / np.sqrt(3.0) / a0
    q0 = M * 2.0 * np.sin(0.5 * theta / 360.0 * 2.0 * np.pi)  # q0 in the unit of A
    aM = a0 / (2.0 * np.sin(0.5 * theta / 360.0 * 2.0 * np.pi))
    Omega0 = aM * aM * 0.5 * np.sqrt(3.0)  # unit cell area in the unit of A^2
    Lat_Para = {'theta': theta, 'a0': a0, 'aM': aM, 'q0': q0, 'Omega0': Omega0}



    para = defaultdict(np.float64)

    para['w2'] = w2

    para['ww2'] = ww2
    para['ww3'] = ww3
    para['ww4'] = ww4
    para['ww6'] = ww6
    para['ww7'] = ww7
    para['ww9'] = ww9
    para['ww10'] = ww10
    para['ww12'] = ww12

    # para['w1'] = w1
    # para['w2'] = w2
    #
    # para['ww1'] = ww1
    # para['ww2'] = ww2
    # para['ww3'] = ww3
    # para['ww4'] = ww4
    # para['ww5'] = ww5
    # para['ww7'] = ww7
    # para['ww10'] = ww10

    para['mx'] = mx
    para['my'] = my
    para['if_su2'] = True
    para['if_zero_twist_kinetic'] = True

    Lat_Para_dict[(comp, stack, theta)] = [Lat_Para, para]

    filename_pre = comp + '_' + str(theta) + '_' + stack + '_'
    Lat_Para, para = Lat_Para_dict[(comp, stack, theta)]

    file_name = folder + filename_pre
    print('Start processing ' + filename_pre)

    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.isfile(file_name):
        os.remove(file_name)

    disp = Mtwist(para, Lat_Para['q0'], theta, which_model = stack)
    disp.update_Qlat(18, 8.8)

    Nk = 12
    disp.kbz = get_kbz_2d(Nk) @ disp.bmat

    hk = disp.get_hk(disp.kbz)
    eg, egv = np.linalg.eigh(hk)
    nk = np.round(np.sqrt(egv.shape[0])).astype(np.int32)
    vk = np.reshape(egv[:, :, 0], (nk, nk, -1))
    vk2 = np.reshape(egv[:, :, 1], (nk, nk, -1))

    vk_smooth, WC = smooth_gauge(vk, disp.kbz, disp.Q_tot, disp.bmat, disp.amat)
    vk2_smooth, WC2 = smooth_gauge(vk2, disp.kbz, disp.Q_tot, disp.bmat, disp.amat)

    WC_shift = [0.0 * disp.amat[1], disp.amat[1], -disp.amat[0]]
    WC_v0 = WC @ disp.amat + WC_shift[0]
    WC_v1 = c3z @ WC_v0 + WC_shift[1]
    WC_v2 = c3z @ c3z @ WC_v0 + WC_shift[2]  # + disp.amat[1]
    WClist = np.asarray([WC_v0, WC_v1, WC_v2])

    WC_shift2 = [disp.amat[1], disp.amat[1], - disp.amat[0]]
    WC_v0 = WC2 @ disp.amat + WC_shift2[0]
    WC_v1 = c3z @ WC_v0 + WC_shift2[1]
    WC_v2 = c3z @ c3z @ WC_v0 + WC_shift2[2]  # + disp.amat[1]
    WC2list = np.asarray([WC_v0, WC_v1, WC_v2])

    fig, ax = plt.subplots()
    ax.scatter(WClist[:, 0], WClist[:, 1])
    ax.scatter(WC2list[:, 0], WC2list[:, 1])
    get_unitcell_boundary(np.linalg.norm(disp.amat[0]), ax)
    ax.set_aspect('equal', adjustable='box')
    plt.title('theta=' + str(theta))
    plt.show()

    # # Get hopping of orb
    #
    # hr_dict = {}
    # hk_eff = np.einsum('ki, kij, kj -> k', np.conj(vk_smooth), hk, vk_smooth)
    # hk_eff_2 = np.einsum('ki, kij, kj -> k', np.conj(vk2_smooth), hk, vk2_smooth)
    #
    # klist = disp.kbz @ np.linalg.inv(disp.bmat)
    # Rlist = []
    # N = 4
    # cutoff = 3.0
    # for i in range(-N, N + 1):
    #     for j in range(-N, N + 1):
    #         r = np.asarray([i, j]) @ disp.amat
    #         if np.linalg.norm(r) < cutoff * np.linalg.norm(disp.amat[0]):
    #             Rlist.append([i, j])
    # Rlist = np.asarray(Rlist)
    #
    # hop = get_hop_from_hk(Rlist, hk_eff, klist)
    # hop2 = get_hop_from_hk(Rlist, hk_eff_2, klist)
    #
    # hr_dict['WC'] = WClist
    # hr_dict['WC2'] = WC2list
    # hr_dict['R_hopping'] = Rlist
    # hr_dict['orb1_t'] = np.copy(hop)
    # hr_dict['orb2_t'] = np.copy(hop2)
    #
    #
    #
    # vk_allval = get_vk_all_wal(disp, vk_smooth, WC_shift)
    # vk2_allval = get_vk_all_wal(disp, vk2_smooth, WC_shift)
    #
    # q = np.copy(disp.kbz)
    # cutoff = 5
    # NG = 5
    # G = []
    # for i in range(-NG, NG + 1):
    #     for j in range(-NG, NG + 1):
    #         Git = i * disp.bmat[0] + j * disp.bmat[1]
    #         if (np.linalg.norm(Git) < cutoff * np.linalg.norm(disp.bmat[0])):
    #             G.append(Git)
    # G = np.asarray(G)
    #
    # form_allval = []
    #
    # print('Calculating ' + filename_pre + 'form factor')
    # for iv in range(0, len(vk_allval)):
    #     vlist = [vk_allval[iv], vk2_allval[iv]]
    #     for i in range(0, len(vlist)):
    #         for j in range(0, len(vlist)):
    #             print('iv, i, j = ', iv, i, j)
    #             form = get_Mform_factor(disp.kbz, vlist[i], vlist[j], q, G, disp)
    #             form_allval.append(form)
    #
    # form_allval = np.asarray(form_allval)  # nvalx(norbxnorb), nk, nq, nG
    #
    # data_dict = {'q': q, 'k': disp.kbz, 'G': G, 'FF_allval': form_allval}
    #
    # kbz = disp.kbz
    #
    #
    # print(filename_pre + ' form factor done')
    #
    # formall = [[], []]
    # it = 0
    # for i in range(0, 3):
    #     for i1 in range(0, 2):
    #         for i2 in range(0, 2):
    #             form = form_allval[it]
    #             it = it + 1
    #             for band in range(0, 2):
    #                 if i1 == band and i2 == band:
    #                     formall[band].append(form)
    # formall = np.asarray(formall)
    #
    # amat = disp.amat
    # cut_R = 1.1 * np.linalg.norm(amat[0])
    # cut_dr = 1.1 * np.linalg.norm(amat[0])
    # N = 5
    # R_list = []
    # dr_list = []
    # for i in range(-N, N + 1):
    #     for j in range(-N, N + 1):
    #         R = np.asarray([i, j]) @ amat
    #         dr = np.asarray([i, j]) @ amat
    #
    #         if np.linalg.norm(R) < cut_R:
    #             R_list.append(R)
    #         if np.linalg.norm(dr) < cut_dr:
    #             dr_list.append(dr)
    # R_list = np.asarray(R_list)
    # dr_list = np.asarray(dr_list)
    #
    #
    # V_R_all = []
    # Vq0_all = []
    # print('Calculating ' + filename_pre + 'interaction')
    # for ij in range(0, len(formall)):
    #     for ijp in range(0, len(formall)):
    #         print('----------------------')
    #         print('ijp', ijp, 'ij', ij)
    #         V_R_allval = []
    #         Vq0_allval = []
    #         qt = np.reshape(q, (-1, 1, 1, 2))
    #         indq0 = np.argmin(np.linalg.norm(q, axis=1))
    #         for i in range(0, len(formall[ij])):
    #             for j in range(0, len(formall[ijp])):
    #                 k1 = np.reshape(kbz, (1, -1, 1, 2))
    #                 k2 = np.reshape(kbz, (1, 1, -1, 2))
    #
    #                 print('ij = ', i, j)
    #                 save_name = file_name + 'orbital_%d_%d_V_tilde_%d_%d.npy' % (ij, ijp, i, j)
    #                 V_val = get_tilde_V_mom_space(disp, kbz, kbz,
    #                                               q, G, formall[ij][i], formall[ijp][j])  # read V for this val
    #                 np.save(save_name, V_val)
    #                 Vq0 = np.mean(V_val[indq0])
    #                 Vq0_allval.append(Vq0)
    #                 V_R = np.zeros((len(R_list), len(dr_list), len(dr_list)), dtype=np.complex128)
    #
    #                 for iR, R in enumerate(R_list):
    #                     for ir1, dr1 in enumerate(dr_list):
    #                         for ir2, dr2 in enumerate(dr_list):
    #                             form_fact = np.exp(1j * k1 @ dr1 - 1j * k2 @ dr2 + 1j * qt @ (R))
    #                             V_R[iR, ir1, ir2] = np.sum(form_fact * V_val) / len(kbz) / len(kbz) / len(q)
    #                 V_R_allval.append(V_R)
    #         V_R_allval = np.asarray(V_R_allval)
    #         V_R_all.append(V_R_allval)
    #         Vq0_allval = np.asarray(Vq0_allval)
    #         Vq0_all.append(Vq0_allval)
    #
    # V_R_all = np.asarray(V_R_all)
    # Vq0_all = np.asarray(Vq0_all)
    # iR0 = np.argmin(np.linalg.norm(R_list, axis=1))
    # idr0 = np.argmin(np.linalg.norm(dr_list, axis=1))
    #
    # VR0 = V_R_all[:,:, iR0, idr0, idr0]
    # Vq0_all = Vq0_all - VR0
    #
    # with h5py.File(file_name, 'w') as h:
    #     for k, v in data_dict.items():
    #         h.create_dataset(k, data=v)
    #     for k, v in hr_dict.items():
    #         h.create_dataset(k, data=v)
    #     h.create_dataset('R', data=R_list)
    #     h.create_dataset('dr', data=dr_list)
    #     h.create_dataset('VR', data=V_R_all)
    #     h.create_dataset('tilde_Vq0', data=Vq0_all)
    # print(filename_pre + ' interaction done')


def read_interaction(comp, stack, theta, folder='WANNIER_PARA/'):
    filename_pre = comp + '_' + str(theta) + '_' + stack + '_'
    file_name = folder + filename_pre
    with h5py.File(file_name, 'r') as h5_file:
        print(list(h5_file.keys()))
        kbz = np.asarray(h5_file['k'])
        q = np.asarray(h5_file['q'])
        G = np.asarray(h5_file['G'])
        formall_allorb = np.asarray(h5_file['FF_allval'])
        R_list = np.asarray(h5_file['R'])
        dr_list = np.asarray(h5_file['dr'])
        V_R_all = np.asarray(h5_file['VR'])
        Vq0_all = np.asarray(h5_file['tilde_Vq0'])
        WC = np.asarray(h5_file['WC'])
        WC2 = np.asarray(h5_file['WC2'])
        R_hopping = np.asarray(h5_file['R_hopping'])
        orb1_t = np.asarray(h5_file['orb1_t'])
        orb2_t = np.asarray(h5_file['orb2_t'])
    print(WC)
    print(WC2)


    bmat = np.asarray([[2.0, 0.0], [1.0, np.sqrt(3.0)]])

    b3d = np.zeros((3, 3))
    b3d[0:2, 0:2] = bmat
    b3d[2, 2] = 2.0 * np.pi
    avec = avec_to_bvec(b3d)
    amat = avec[0:2, 0:2]
    Rlist_cart = R_hopping @ amat

    fig, ax = plt.subplots()
    ax.scatter(WC[:, 0], WC[:, 1])
    ax.scatter(WC2[:, 0], WC2[:, 1])
    get_unitcell_boundary(np.linalg.norm(amat[0]), ax)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    ind_ = np.argmin(np.linalg.norm(Rlist_cart, axis=1))
    print('Onsite = ', orb1_t[ind_])
    orb1_t[ind_] = 0.0  # set onsite to be zero

    fig, ax = plt.subplots()
    im = ax.scatter(Rlist_cart[:, 0], Rlist_cart[:, 1], c=np.abs(orb1_t))
    plt.colorbar(im, ax=ax)
    plt.ylabel('Rx')
    plt.xlabel('Ry')
    plt.title('hopping strength')
    plt.show()

    ind_ = np.argmin(np.linalg.norm(Rlist_cart, axis=1))
    print('Onsite = ', orb2_t[ind_])
    orb2_t[ind_] = 0.0  # set onsite to be zero
    fig, ax = plt.subplots()
    im = ax.scatter(Rlist_cart[:, 0], Rlist_cart[:, 1], c=np.abs(orb2_t))
    plt.colorbar(im, ax=ax)
    plt.ylabel('Rx')
    plt.xlabel('Ry')
    plt.title('hopping strength')
    plt.show()



    for ind in range(0, len(V_R_all)):
        print('-------------------------------')
        print('orbital %d and %d ' % (int(ind/2), ind % 2))
        V_R_allval = V_R_all[ind]
        val_map = {}
        it = 0
        for i in range(0,3):
            for j in range(0, 3):
                val_map[it] = (i, j)
                it += 1

        for id_ in range(0, len(V_R_allval)):
            for iR, R in enumerate(R_list):
                for ir1, dr1 in enumerate(dr_list):
                    for ir2, dr2 in enumerate(dr_list):
                        v = V_R_allval[id_, iR, ir1, ir2]
                        if (np.abs(v) > 50):
                            print(val_map[id_], R, dr1, dr2, np.round(v, 3))

        ind_off = [True for i in range(0, len(dr_list))]
        ind_ = np.argmin(np.linalg.norm(dr_list, axis=1))
        ind_off[ind_] = False
        ind_onsite = np.logical_not(ind_off)  # index of on-site

        V_on = V_R_allval[:, :, ind_onsite][:, :, :, ind_onsite]
        V_off_on = V_R_allval[:, :, ind_off][:, :, :, ind_onsite]
        V_on_off = V_R_allval[:, :, ind_onsite][:, :, :, ind_off]
        V_off_off = V_R_allval[:, :, ind_off][:, :, :, ind_off]

        V_list = [V_on, V_off_on, V_on_off, V_off_off]
        for V in V_list:
            plt.plot(np.max(np.abs(V), axis=(0, 2, 3)), 'o')
        # plt.ylim(0,10)
        plt.show()

        for i in range(0, 3):
            for j in range(0, 3):
                print('valley %d and %d ' % (i, j))
                u = V_R_allval[i * 3 + j]
                u1d = np.flip(np.sort(np.reshape(np.abs(u), -1)))

                indr0 = np.argmin(np.linalg.norm(dr_list, axis=1))
                unn = u[:, indr0, indr0]
                unn1d = np.flip(np.sort(np.reshape(np.abs(unn), -1)))
                if (np.max(np.abs(unn1d[0:7] - u1d[0:7])) > 5):
                    print('Max int = ', u1d[0:10])
                    print('Max nn int = ', unn1d[0:10])

                print("NN int = ")
                print(np.round(unn1d, 3))



if __name__ == '__main__':
    comp = 'SnSe2'
    stack = 'AB'
    theta = 3.89
    # run_parameter(comp, stack)
    run_parameter(comp, stack)





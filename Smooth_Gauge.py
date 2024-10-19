import numpy as np
import matplotlib.pyplot as plt


def get_embed(Qtot, bshift):
    M_embed = np.zeros((len(Qtot), len(Qtot)), dtype=np.complex128)

    for i in range(0, len(Qtot)):
        for j in range(0, len(Qtot)):

            if (np.linalg.norm(Qtot[i] + bshift - Qtot[j]) < 0.01 * np.linalg.norm(bshift)):
                M_embed[j, i] = 1.0

    return M_embed


def smooth_gauge(vk, kbz, Qtot, bmat, amat):
    N1, N2 = vk.shape[0], vk.shape[1]
    vk_smooth = np.zeros_like(vk)

    vk_smooth[0, 0] = vk[0, 0]
    for i in range(0, N2 - 1):
        vk_smooth[0, i + 1] = vk[0, i + 1]
        phi = np.angle(np.conj(vk_smooth[0, i]) @ vk_smooth[0, i + 1])
        vk_smooth[0, i + 1] = vk_smooth[0, i + 1] * np.exp(-1j * phi)

    M_embed = get_embed(Qtot, bmat[1])
    phase = (np.conj(vk_smooth[0, -1]) @ M_embed @ vk_smooth[0, 0])
    n_2pi = np.round(np.angle(phase) / 2 / np.pi)
    phase = np.angle(phase) - n_2pi * 2.0 * np.pi

    for i in range(0, N2):
        vk_smooth[0, i] = vk_smooth[0, i] * np.exp(1j * phase  * i / (N2))
    # print('phase factor along k1=0/2/pi', np.angle(phase) / 2 / np.pi, 'mod = ', np.abs(phase))

    phase_list = []
    M_embed = get_embed(Qtot, bmat[0])
    for j in range(0, N2):

        for i in range(0, N1 - 1):
            vk_smooth[i + 1, j] = vk[i + 1, j]
            phi = np.angle(np.conj(vk_smooth[i, j]) @ vk_smooth[i + 1, j])
            #             phi = np.angle(phase)
            vk_smooth[i + 1, j] = vk_smooth[i + 1, j] * np.exp(-1j * phi)

        phase = (np.conj(vk_smooth[-1, j]) @ M_embed @ vk_smooth[0, j])
        n_2pi = np.round(np.angle(phase) / 2 / np.pi)
        phi = np.angle(phase) - n_2pi * 2.0 * np.pi
        phase_list.append(phi)
        for i in range(0, N1):
            vk_smooth[i, j] = vk_smooth[i, j] * np.exp(1j * i * phi/ (N1))

    phase_der = np.zeros((N1, N2, 2), dtype=np.complex128)
    abs_der = np.zeros((N1, N2, 2), dtype=np.complex128)
    for i in range(0, N1):
        for j in range(0, N2):
            i2 = i + 1
            j2 = j
            if (i2 == N1):
                M_embed = get_embed(Qtot, bmat[0])
                ov = (np.conj(vk_smooth[i, j]) @ M_embed @ vk_smooth[i2 % N1, j2 % N2])
            else:
                ov = np.conj(vk_smooth[i, j]) @ vk_smooth[i2, j2]

            abs_der[i, j, 0] = np.abs(ov)
            phase_der[i, j, 0] = np.angle(ov)

            i2 = i
            j2 = j + 1
            if (j2 == N1):
                M_embed = get_embed(Qtot, bmat[1])
                ov = (np.conj(vk_smooth[i, j]) @ M_embed @ vk_smooth[i2 % N1, j2 % N2])
            else:
                ov = np.conj(vk_smooth[i, j]) @ vk_smooth[i2, j2]

            abs_der[i, j, 1] = np.abs(ov)
            phase_der[i, j, 1] = np.angle(ov)

    phase = - np.mean(phase_der, axis=(0, 1))  # get average phase diff, which gives Wannier center
    print(phase)
    WC = np.real(phase) * np.asarray([N1, N2]) / 2.0 / np.pi
    print("wannier center =%.3f am1 + %.3f am2" % (WC[0], WC[1]))

    if (True):
        print('check <vk| vk+dk> ')
        fig, axs = plt.subplots(ncols=2, figsize=(14, 6))
        plt.suptitle('|<vk|vk+dk>|')
        axs[0].set_title('x')
        axs[1].set_title('y')

        im = axs[0].scatter(kbz[:, 0], kbz[:, 1], c=abs_der[:, :, 0], s=10)
        fig.colorbar(im, ax=axs[0])
        im = axs[1].scatter(kbz[:, 0], kbz[:, 1], c=abs_der[:, :, 1], s=10)
        fig.colorbar(im, ax=axs[1])
        plt.show()

        fig, axs = plt.subplots(ncols=2, figsize=(14, 6))
        plt.suptitle('arg(<vk|vk+dk>)')
        axs[0].set_title('x')
        axs[1].set_title('y')

        im = axs[0].scatter(kbz[:, 0], kbz[:, 1], c=np.real(phase_der[:, :, 0]), s=10)  # , vmax=0.2, vmin =-0.2)
        fig.colorbar(im, ax=axs[0])
        im = axs[1].scatter(kbz[:, 0], kbz[:, 1], c=np.real(phase_der[:, :, 1]), s=10)  # , vmax=0.2, vmin =-0.2)
        fig.colorbar(im, ax=axs[1])
        plt.show()

    return np.reshape(vk_smooth, (N1 * N2, -1)), WC

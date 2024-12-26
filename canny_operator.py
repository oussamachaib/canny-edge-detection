import numpy as np

class canny:
    '''
    Canny edge operator

    Computes a binary edge corresponding to local spatial gradient maxima in the image.

    Parameters
    __________
    sigma: Standard deviation of the Gaussian kernel (float)
    k: Kernel width (int - k = 6*sigma-1 by default)
    t_lo: Lower threshold (float between 0 and 1, and lower or equal to t_hi - t_lo = 0.25 by default)
    t_hi: Upper threshold (float between 0 and 1, and greater or equal to t_lo - t_hi = 0.5 by default)

    Returns
    _______
    '''

    def __init__(self, sigma, k = None, t_lo = 0.25, t_hi = 0.5):
        self.sigma = sigma
        self.k = k
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.H = None

    def _DoG1(self):
        '''
        1D Derivative of Gaussian (DoG) kernel.
        The kernel is horizontal by definition with dimensions (1,k).
        If no value of k is provided, it is automatically set to: k = 6*sigma - 1

        Parameters
        __________
        sigma: Standard deviation of the Gaussian kernel (float)
        k: Kernel width (int)

        Returns
        _______
        H: 1D Gaussian kernel (1,k)
        '''

        if self.k == None:
            self.k = 3 * 2 * int(self.sigma) - 1

        H = np.zeros((1, self.k))
        center = int(self.k / 2)

        for i in range(self.k):
            x = i - center
            H[0, i] = (-x / self.sigma ** 2) * (1 / (np.sqrt(2 * np.pi * self.sigma ** 2))) * (np.exp(-x ** 2 / (2 * self.sigma ** 2)))

        H /= np.sum(H)

        return H

    def _convolve(self, I, H):
        '''
        Computes the convolution of the input image (I) with kernel (H).
        Reflection padding is used to accomodate convolutional operations near the edges of the image.

        Parameters
        __________
        I: Input image (2D uint8 array)
        H: Kernel (-)

        Returns
        _______
        F: Convolution of image with kernel.

        '''
        delta_y = int(H.shape[0] / 2)
        delta_x = int(H.shape[1] / 2)

        I_pad = np.zeros((I.shape[0] + delta_y * 2, I.shape[1] + delta_x * 2))
        I_pad[delta_y:(I.shape[0] + delta_y), delta_x:(I.shape[1] + delta_x)] = I

        F_pad = np.zeros((I_pad.shape))

        # Mirroring to remove edge effects
        # x
        if delta_x > 0:
            I_pad[:, 0:delta_x] = np.flip(I_pad[:, delta_x:2 * delta_x], axis=1)
            I_pad[:, -delta_x:] = np.flip(I_pad[:, -2 * delta_x:-delta_x], axis=1)
        # y
        if delta_y > 0:
            I_pad[0:delta_y, :] = np.flip(I_pad[delta_y:2 * delta_y, :], axis=0)
            I_pad[-delta_y:, :] = np.flip(I_pad[-2 * delta_y:-delta_y, :], axis=0)

        for i in range(delta_y, I.shape[0] + delta_y):
            for j in range(delta_x, I.shape[1] + delta_x):
                F_pad[i, j] = np.sum(I_pad[(i - delta_y):(i + delta_y + 1), (j - delta_x):(j + delta_x + 1)] * H)

        F = F_pad[delta_y:(I.shape[0] + delta_y), delta_x:(I.shape[1] + delta_x)]

        return F

    def _compute_gradient(self, X):
        '''
        Computes the derivative of the image using the precomputed 1D DoG kernel.
        This is done using two 1D convolution for simplicity.

        Parameters
        __________
        X: Image (2D uint8 array)

        Returns
        _______
        Gx: Horizontal gradient (2D float array)
        Gy: Vertical gradient (2D float array)

        '''

        DoG = self._DoG1()
        Gx = self._convolve(X, DoG)
        Gy = self._convolve(X, DoG.T)
        return Gx, Gy

    def _get_orientation(self, Gx, Gy):
        '''
        Computes the orientation of each pixel using its corresponding vertical and horizontal gradient values.
        The computational trick discussed in Burger and Burge's book (Digital Image Processing: An Algorithmic Introduction Using Java, 2008) is used to get around angle computation
        Orientation classes (quadrant labels) : 0 (left - right), 1 (top left - bottom right), 2 (top - bottom), 3 (top right - bottom left)

        Parameters
        __________
        Gx: Horizontal gradient (2D float array)
        Gy: Vertical gradient (2D float array)

        Returns
        _______
        orientation: Quadrant values between 0 and 3 (2D int array)
        '''

        # Computing gradient
        G = np.sqrt(Gx ** 2 + Gy ** 2)
        G /= G.max()

        # Computational trick: by rotation the octants and clever use of mirroring, we can limit the analysis to a single quadrant, and without the need for angle computation via arctan2
        angle_rot = np.pi / 8

        # Rotating by pi/8 clockwise -- bear in mind the image's y-axis points downwards!
        Gx_rot = Gx * np.cos(angle_rot) - Gy * np.sin(angle_rot)
        Gy_rot = Gx * np.sin(angle_rot) + Gy * np.cos(angle_rot)

        # Mirroring octants to upper quadrants
        Gx_rot[Gy_rot < 0] = - Gx_rot[Gy_rot < 0]
        Gy_rot[Gy_rot < 0] = - Gy_rot[Gy_rot < 0]

        # Orientation assignment (quadrants numbered 0-3)
        orientation = np.zeros((G.shape))
        orientation[(Gx_rot >= 0) & (Gy_rot > Gx_rot)] = 1
        orientation[(Gx_rot < 0) & (Gy_rot > -Gx_rot)] = 2
        orientation[(Gx_rot < 0) & (-Gx_rot >= Gy_rot)] = 3

        return orientation

    def _nms(self, G, orientation):
        '''
        Non-maxima suppression

        Thins the input gradient magnitude array by preserving local gradient magnitude maxima in the direction given by the orientation class (0-3).
        Pixels are scanned via a 3x3 kernel and their magnitudes are compared to the pixels in their respective 8-neighborhoods.

        Parameters
        __________
        G: Gradient magnitude (2D float array)
        orientation: Quadrant values between 0 and 3 (2D int array)

        Returns
        _______
        E_nms: Thinned gradient map (2D 1/0's array)
        '''

        # Mirror/reflection padding
        G_pad = np.zeros((G.shape[0] + 2, G.shape[1] + 2))
        G_pad[1:(G.shape[0] + 1), 1:(G.shape[1] + 1)] = G

        # Declaring placeholder
        E_nms = np.ones(G.shape)

        for i in range(1, G.shape[0]):
            for j in range(1, G.shape[1]):
                # Scanning each value
                K = G_pad[i - 1:i + 2, j - 1:j + 2]
                if K[1, 1] < self.t_lo:
                    E_nms[i - 1, j - 1] = False
                elif orientation[i, j] == 0:
                    if np.argmax(K[1, :]) != 1:
                        E_nms[i - 1, j - 1] = 0
                elif orientation[i, j] == 1:
                    if np.argmax(np.diag(K)) != 1:
                        E_nms[i - 1, j - 1] = 0
                elif orientation[i, j] == 2:
                    if np.argmax(K[:, 1]) != 1:
                        E_nms[i - 1, j - 1] = 0
                else:
                    if np.argmax(np.diag(np.fliplr(K))) != 1:
                        E_nms[i - 1, j - 1] = 0
        return E_nms

    def _hystt(self, G, E_nms):
        '''
        Hysteresis thresholding

        Edge pixels obtained after NMS are preserved if (1) their gradient magnitude is higher than the provided upper threshold (t_hi),
        OR (2) their gradient magnitude is between the upper and lower thresholds (t_hi and t_low, respectively) AND they are 8-connected
        to a pixel meeting condition (1).

        Parameters
        __________
        G: Gradient magnitude (2D float array)
        E_nms: Thinned gradient map (2D 1/0's array)

        Returns
        E: Object edge (2D 1/0's array)
        _______
        '''

        E = (G * E_nms) > self.t_lo
        args = np.argwhere(E)

        for i in range(len(args)):
            # Scanning each value
            y1 = np.max((0, args[i, 0] - 1))
            y2 = np.min((E_nms.shape[0], args[i, 0] + 2))
            x1 = np.max((0, args[i, 1] - 1))
            x2 = np.min((E_nms.shape[1], args[i, 1] + 2))

            if np.max(G[y1:y2, x1:x2]) < self.t_hi:
                E[args[i, 0], args[i, 1]] = 0

        return E

    def fit_predict(self, X):
        '''
        Running the Canny edge detection algorithm.

        Parameters
        __________
        X: Image (2D uint8 array)

        Returns
        E: Object edge (2D 1/0's array)
        _______
        '''
        Gx, Gy = self._compute_gradient(X)
        alpha = self._get_orientation(Gx,Gy)

        G = np.sqrt(Gx**2 + Gy**2)
        G /= G.max()

        E_nms = self._nms(G, alpha)
        E = self._hystt(G, E_nms)

        return E

    def _thicken(self, E, H = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])/5, n = 1):
        '''
        Thickening 1D edges by repeated convolutions (n times) with the provided kernel H.

        Parameters
        __________
        E: Object edge (2D 1/0's array)
        H: Kernel (3x3 cross kernel by default)
        n: Number of convolution iterations (one by default)

        Returns
        E_thick: Thickened object edge (2D 1/0's array)
        _______
        '''
        E_thick = E.copy()
        for i in range(n):
            E_thick = self._convolve(E_thick, H)
        return E_thick/E_thick.max()


#%% Testing snippet
import cv2
import matplotlib.pyplot as plt
import time

plt.rcParams['text.usetex'] = True # TeX font
plt.rcParams['font.family'] = 'serif' # Serif type
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' # AMS-LaTeX package

t_lo = 0.05
t_hi = .2
sigma = 2

# Loading image
species = ['daisy', 'rose', 'lotus', 'violet']

fig = plt.figure(figsize = (10,4.5), dpi = 300)

for i, j in enumerate(species):
    # Reading image
    I = cv2.imread(f"demo_images/{j}.png", cv2.IMREAD_GRAYSCALE)

    # Running model
    start = time.time()
    model = canny(sigma=sigma, t_lo=t_lo, t_hi=t_hi)
    E = model.fit_predict(I)
    print(f'[Image {i}] Computation time: {time.time() - start:.2} (s)')
    E = model._thicken(E,n = 5)

    ax = fig.add_subplot(2,len(species),i+1)
    ax.imshow(I, vmin = 0, vmax = 255, cmap = 'gray', alpha = 1)
    ax.set_title(fr'{j}', fontsize = 20)
    ax.axis('off')

    ax = fig.add_subplot(2,len(species),i+1+4)
    ax.imshow(I, vmin=0, vmax=255, cmap='gray', alpha=1)
    ax.imshow(E, vmin = 0, vmax = 1, cmap = 'Reds', alpha = .5)
    ax.axis('off')



plt.tight_layout()
plt.show()

#%% Testing computation time of cv2

# Running model
start = time.time()
E = cv2.Canny(I, threshold1=t_lo*255, threshold2=t_hi*255)
print(f'Computation time: {time.time() - start:.2} (s)')


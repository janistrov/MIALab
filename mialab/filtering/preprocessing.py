"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.interpolate import interp1d
from skfuzzy import cmeans
from scipy.signal import argrelmax
import statsmodels.api as sm
import os


class ImageNormalization(pymia_fltr.IFilter):
    """Represents a normalization filter."""

    def __init__(self, id_, T_, norm_method='no', standard_scale=None, percs=None, mask=None):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()
        # general
        self.id_ = id_
        self.T_ = T_
        self.norm_method = norm_method
        self.mask = mask
        # histogram matching
        self.standard_scale = standard_scale
        self.percs = percs

    def execute(self, image: sitk.Image, params: pymia_fltr.IFilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # STUDENT: implementation of normalization
        if self.norm_method == 'z':
            print('Normalization method: Z-Score')
            mask = sitk.GetArrayFromImage(self.mask)
            mean = img_arr[mask == 1].mean()
            std = img_arr[mask == 1].std()
            img_arr = (img_arr - mean) / std

        elif self.norm_method == 'ws':
            print('Normalization method: White Stripe')
            indices = self.white_stripe(img_arr)
            plt.figure()
            plt.title('White Stripe Mask of of ID ' + self.id_)
            plt.imshow(indices[100, :, :])
            plt.axis('off')
            plt.savefig('./mia-result/plots/WS_Mask_' + self.T_ + '_' + self.id_ + '.png')
            plt.close()
            # Normalization step
            mean = np.mean(img_arr[indices])
            std = np.std(img_arr[indices])
            img_arr = (img_arr - mean) / std

        elif self.norm_method == 'hm':
            print('Normalization method: Histogram Matching')
            # self.save_hist(img_arr)  # optional for inspection
            img_arr = self.do_hist_norm(img_arr)
            # self.save_hist(img_arr, normalized=True)  # optional for inspection

        elif self.norm_method == 'fcm':
            print('Normalization method: FCM White Matter Aligning')
            threshold = 0.6
            fcm_clusters = self.fcm_mask(img_arr, maxiter=30)
            # Create mask with white matter cluster
            if self.T_ is 'T1w':
                wm_mask = fcm_clusters[..., 2] > threshold
            elif self.T_ is 'T2w':
                wm_mask = fcm_clusters[..., 0] > threshold
            else:
                print('Wrong entry for image contrast')
                wm_mask = None
            # Plot clusters for visual inspection
            clusters = np.zeros(img_arr.shape)
            for i in range(3):
                clusters[fcm_clusters[..., i] > threshold] = i + 1
            plt.figure()
            plt.title('White Matter Mask of of ID ' + self.id_)
            plt.imshow(clusters[100, :, :])
            plt.axis('off')
            plt.savefig('./mia-result/plots/WM_Mask_' + self.T_ + '_' + self.id_ + '.png')
            plt.close()
            # Normalization step
            wm_mean = img_arr[wm_mask == 1].mean()
            img_arr = (img_arr/wm_mean)

        elif self.norm_method == 'no':
            print('No Normalization')

        else:
            print('Normalization method not known. Pre processing runs without normalization.')

        # conversion to simpleITK image
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def white_stripe(self, image, width=0.05):
        """
        Find the "(normal appearing) white (matter) stripe" of the input MR image
        and return the indices

        Args:
            image (ndarray): The image to normalize
            width (float): Width quantile for the "white (matter) stripe"

        Returns:
            ws_ind (array): The white stripe indices (boolean mask)
        """
        brain_mask = sitk.GetArrayFromImage(self.mask)
        voi = image[brain_mask == 1]
        if self.T_ is 'T1w':
            mode, grid, pdf = self.hist_get_last_mode(voi)
        elif self.T_ is 'T2w':
            mode, grid, pdf = self.hist_get_largest_mode(voi)
        else:
            print('Wrong entry for image contrast')
            mode, grid, pdf = None, None, None
        img_mode_q = np.mean(voi < mode)
        ws = np.percentile(voi, (max(img_mode_q - width, 0) * 100, min(img_mode_q + width, 1) * 100))
        ws_ind = np.logical_and(image > ws[0], image < ws[1])
        if len(ws_ind) == 0:
            print('WhiteStripe failed to find any valid indices!')
        plt.figure()
        plt.title('White Stripe of of ID ' + self.id_)
        plt.xlabel('Intensity')
        plt.ylabel('PDF')
        plt.plot(grid, pdf, color='b')
        plt.plot([ws[0], ws[0]], [0, max(pdf)], '--', color='r')
        plt.plot([ws[1], ws[1]], [0, max(pdf)], '--', color='r')
        plt.savefig('./mia-result/plots/WS_' + self.T_ + '_' + self.id_ + '.png')
        plt.close()

        return ws_ind

    def hist_get_last_mode(self, voi):
        """
        Gets the largest (reliable) peak in the histogram

        Args:
            voi (ndarray): Voxels of interest

        Returns:
            last_peak (int): Index of the largest peak
            grid (array): Domain of the pdf
            pdf (array): Kernel density estimate of the pdf of data
        """
        rare_prop = 96
        rare_thresh = np.percentile(voi, rare_prop)  # intensity corresponding to 96 quantile
        which_rare = voi >= rare_thresh
        voi = voi[which_rare == 0]  # cuts away intensities above rare_thresh
        grid, pdf = self.smooth_hist(voi)
        maxima = argrelmax(pdf)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
        last_peak = grid[maxima[-1]]  # gives intensity (read on grid) of last pdf peak

        return last_peak, grid, pdf

    def hist_get_largest_mode(self, data):
        """
        Gets the last (reliable) peak in the histogram

        Args:
            data (np.array): Voxels of interest

        Returns:
            largest_peak (int): Index of the largest peak
        """
        grid, pdf = self.smooth_hist(data)
        largest_peak = grid[np.argmax(pdf)]
        return largest_peak, grid, pdf

    def smooth_hist(self, data):
        """
        Use KDE to get smooth estimate of histogram

        Args:
            data (np.array): Voxels of interest

        Returns:
            grid (array): Domain of the pdf
            pdf (array): Kernel density estimate of the pdf of data
        """
        data = data.flatten().astype(np.float64)
        bw = data.max() / 80
        kde = sm.nonparametric.KDEUnivariate(data)
        kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
        pdf = 100.0 * kde.density
        grid = kde.support

        return grid, pdf

    def do_hist_norm(self, image):
        """
        Does the Nyul and Udupa histogram normalization routine with a given set of learned landmarks

        Args:
            image (np.array): The image to normalize

        Returns:
            image_norm (array): Normalized image
        """
        idx = (sitk.GetArrayFromImage(self.mask) == 1)
        masked = image[idx]
        landmarks = np.percentile(masked, self.percs)
        f = interp1d(landmarks, self.standard_scale, fill_value='extrapolate')
        normed = f(image)

        return normed

    def save_hist(self, image, normalized=False):
        """
        Makes a intensity histogram and saves it

        Args:
            image (np.array): The image
            normalized (bool): Indicates if the image is normalized (True) or not (False)
        """
        idx = (sitk.GetArrayFromImage(self.mask) == 1)
        plt.hist(np.ravel(image[idx]), bins=200)
        if normalized is False:
            plt.title('Intensity Histogram of ID ' + self.id_)
            plt.savefig('./mia-result/plots/Histogram_' + self.T_ + '_' + self.id_ + '.png')
        else:
            plt.title('Normalized Intensity Histogram of ID ' + self.id_)
            plt.savefig('./mia-result/plots/Histogram_' + self.T_ + '_' + self.id_ + '_Normalized.png')
        plt.close()

    def fcm_mask(self, image, maxiter=50):
        """
        creates a mask of tissue classes for a target brain with fuzzy c-means

        Args:
            image (np.array): The image
            brain_mask (np.array): The brain mask
            maxiter (scalar): Maximum iterations for fuzzy c-means

        Returns:
            mask (np.ndarray): Membership values for each of three classes in the image
        """
        brain_mask = sitk.GetArrayFromImage(self.mask)
        [cntr, mem, _, _, _, _, fpc] = cmeans(image[brain_mask == 1].reshape(-1, len(image[(brain_mask == 1)])),
                                              3, 2, error=0.005, maxiter=maxiter)
        mem_list = [mem[i] for i, _ in sorted(enumerate(cntr), key=lambda x: x[1])]  # CSF/GM/WM
        mask = np.zeros(image.shape + (3,))
        for i in range(3):
            mask[..., i][brain_mask == 1] = mem_list[i]

        # for inspection of mask on 3D Slicer
        # save_mask = sitk.GetImageFromArray(mask)
        # save = os.path.join('./mia-result/norm images/test_mask.nii.gz')
        # sitk.WriteImage(save_mask, save)

        return mask

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


class SkullStrippingParameters(pymia_fltr.IFilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.IFilter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        mask = params.img_mask  # the brain mask

        # todo: remove the skull from the image by using the brain mask
        # warnings.warn('No skull-stripping implemented. Returning unprocessed image.')
        image = sitk.Mask(image, mask)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.IFilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.IFilter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        # todo: replace this filter by a registration. Registration can be costly, therefore, we provide you the
        # transformation, which you only need to apply to the image!
        # warnings.warn('No registration implemented. Returning unregistered image')

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different
        if is_ground_truth:
            # apply transformation to ground truth and brain mask using nearest neighbor interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkNearestNeighbor, 0,
                                  image.GetPixelIDValue())
        else:
            # apply transformation to T1w and T2w images using linear interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkLinear, 0.0,
                                  image.GetPixelIDValue())

        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)

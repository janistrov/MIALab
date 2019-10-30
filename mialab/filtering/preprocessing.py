"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.interpolate import interp1d
import mialab.utilities.pipeline_utilities as util
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

        # histogram matching
        self.standard_scale = standard_scale
        self.mask = mask
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
            mean = img_arr.mean()
            std = img_arr.std()
            img_arr = (img_arr - mean) / std

        elif self.norm_method == 'hm':
            print('Normalization method: Histogram Matching')
            self.plot_hist(img_arr)
            img_arr = self.do_hist_norm(img_arr)
            self.plot_hist(img_arr, normalized=True)

        elif self.norm_method == 'fcm':
            print('Normalization method: FCM WM-Aligning')
            threshold = 0.8
            brain_mask = sitk.GetArrayFromImage(self.mask)
            fcm_clusters = util.fcm_mask(img_arr, brain_mask, maxiter=50)

            if self.T_ is 'T1w':
                wm_mask = fcm_clusters[..., 2] > threshold
            elif self.T_ is 'T2w':
                wm_mask = fcm_clusters[..., 0] > threshold

            clusters = np.zeros(img_arr.shape)
            for i in range(3):
                clusters[fcm_clusters[..., i] > threshold] = i + 1
            plt.figure()
            plt.title('FCM Masks of ID ' + self.id_)
            plt.imshow(clusters[100, :, :])
            plt.axis('off')
            plt.savefig('./mia-result/plots/FCM_Mask_' + self.T_ + '_' + self.id_ + '.png')
            plt.close()

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

    def do_hist_norm(self, image):
        """
        Does the Nyul and Udupa histogram normalization routine with a given set of learned landmarks

        Args:
            image (array): The image to normalize

        Returns:
            image_norm (array): Normalized image
        """
        idx = (sitk.GetArrayFromImage(self.mask) == 1)
        masked = image[idx]
        landmarks = np.percentile(masked, self.percs)
        f = interp1d(landmarks, self.standard_scale, fill_value='extrapolate')
        normed = f(image)

        return normed

    def plot_hist(self, image, normalized=False):
        """
        Makes a intensity histogram and saves it

        Args:
            image (array): The image
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

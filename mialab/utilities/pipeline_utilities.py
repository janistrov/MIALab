"""This module contains utility classes and functions."""
import enum
import os
import typing as t
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk
from scipy.interpolate import interp1d
import statsmodels.api as sm

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc

atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


# STUDENT: Add artifact to images
def add_artifact(images: structure.BrainImage, artifact_method):
    """Adds artifacts to images.

    Args:
        images (structure.BrainImage): The images
        artifact_method (str): Artifact Method
    """
    print('artifact method: ' + artifact_method)

    img = []
    img.append(sitk.GetArrayFromImage(images.images[structure.BrainImageTypes.T1w]))
    img.append(sitk.GetArrayFromImage(images.images[structure.BrainImageTypes.T2w]))
    mask = sitk.GetArrayFromImage(images.images[structure.BrainImageTypes.BrainMask])

    img_artifact = []

    if artifact_method == 'none':
        for i in range(2):
            img_artifact.append(img[i])

    elif artifact_method == 'gaussian noise':
        for i in range(2):
            img_artifact.append(img[i] + np.random.normal(1.0, 100, img[i].shape))
            # img_artifact[i] = np.where(mask == 0, 0, img_artifact[i])
            # img_artifact[i] = np.clip(img_artifact[i], 0, np.max(img[i]))

    images.images[structure.BrainImageTypes.T1w] = sitk.GetImageFromArray(img_artifact[0])
    images.images[structure.BrainImageTypes.T2w] = sitk.GetImageFromArray(img_artifact[1])


# STUDENT: Plot smooth histogram for inspection2
def get_masked_intensities(image: sitk.Image, mask: sitk.Image):
    """Plots a slice of an image.

    Args:
        image (sitk.Image): The image
        mask (sitk.Image):  The mask
    """
    img_arr = sitk.GetArrayFromImage(image)
    msk_arr = sitk.GetArrayFromImage(mask)
    masked_intensities = img_arr[msk_arr == 1]

    return masked_intensities


# STUDENT: Plot a slice for visual inspection
def plot_slice(image: sitk.Image):
    """Plots a slice of an image.

    Args:
        image (sitk.Image): The image
    """
    img_arr = sitk.GetArrayFromImage(image)
    plt.figure()
    plt.imshow(img_arr[80, :, :], cmap='gray')
    plt.colorbar()
    plt.show()


# STUDENT: Calculate standard histogram for histogram matching norm
def hist_to_match(imgs: list, i_min=1, i_max=99, i_s_min=1,
                  i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """Determine the standard scale for the set of images
    
    Args:
        imgs (list): The images in a structure
        i_min (float): Minimum percentile to consider in the images
        i_max (float): Maximum percentile to consider in the images
        i_s_min (float): Minimum percentile on the standard scale
        i_s_max (float): Maximum percentile on the standard scale
        l_percentile (int): Middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): Middle percentile upper bound (e.g., for deciles 90)
        step (int): step for Middle percentiles (e.g., for deciles 10)

    Returns:
        standard_scales (tuple of array): Average landmark intensity for T1w and T2w images
        percs (array): Array of all percentiles used
    """
    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile + 1, step), [i_max]))
    T1w_standard_scale = np.zeros(len(percs))
    T2w_standard_scale = np.zeros(len(percs))

    for i, image in enumerate(imgs):
        # get images as arrays
        T1w = sitk.GetArrayFromImage(image.images[structure.BrainImageTypes.T1w])
        T2w = sitk.GetArrayFromImage(image.images[structure.BrainImageTypes.T2w])
        mask = sitk.GetArrayFromImage(image.images[structure.BrainImageTypes.BrainMask])

        # get landmarks
        T1w_masked, T2w_masked = T1w[(mask == 1)], T2w[(mask == 1)]
        T1w_landmarks, T2w_landmarks = np.percentile(T1w_masked, percs), np.percentile(T2w_masked, percs)

        # interpolate ends
        T1w_min_p, T2w_min_p = np.percentile(T1w_masked, i_min), np.percentile(T2w_masked, i_min)
        T1w_max_p, T2w_max_p = np.percentile(T1w_masked, i_max), np.percentile(T2w_masked, i_max)
        T1w_f = interp1d([T1w_min_p, T1w_max_p], [i_s_min, i_s_max])
        T2w_f = interp1d([T2w_min_p, T2w_max_p], [i_s_min, i_s_max])
        T1w_landmarks, T2w_landmarks = np.array(T1w_f(T1w_landmarks)), np.array(T2w_f(T2w_landmarks))

        # get standart scale
        T1w_standard_scale += T1w_landmarks
        T2w_standard_scale += T2w_landmarks

    T1w_standard_scale = T1w_standard_scale / len(imgs)
    T2w_standard_scale = T2w_standard_scale / len(imgs)

    return (T1w_standard_scale, T2w_standard_scale), percs


# STUDENT: Load images to get standard scale histogram
def hm_load_images(id_: str, paths: dict) -> structure.BrainImage:
    """Loads an image

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.

    Returns:
        (structure.BrainImage):
    """
    paths_temp = paths.copy()
    path = paths_temp.pop(id_, '')
    path_to_transform = paths_temp.pop(structure.BrainImageTypes.RegistrationTransform, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths_temp.items()}
    transform = sitk.ReadTransform(path_to_transform)
    img = structure.BrainImage(id_, path, img, transform)

    return img


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))
    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images have not the same image properties')


class FeatureImageTypes(enum.Enum):
    """Represents the feature image types."""

    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5


class FeatureExtractor:
    """Represents a feature extractor."""

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        self.coordinates_feature = kwargs.get('coordinates_feature', False)
        self.intensity_feature = kwargs.get('intensity_feature', False)
        self.gradient_intensity_feature = kwargs.get('gradient_intensity_feature', False)

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        # warnings.warn('No features from T2-weighted image extracted.')

        if self.coordinates_feature:
            atlas_coordinates = fltr_feat.AtlasCoordinates()
            self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1w])

        if self.intensity_feature:
            self.img.feature_images[FeatureImageTypes.T1w_INTENSITY] = self.img.images[structure.BrainImageTypes.T1w]
            self.img.feature_images[FeatureImageTypes.T2w_INTENSITY] = self.img.images[structure.BrainImageTypes.T2w]

        if self.gradient_intensity_feature:
            # compute gradient magnitude images
            self.img.feature_images[FeatureImageTypes.T1w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1w])
            self.img.feature_images[FeatureImageTypes.T2w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2w])

        self._generate_feature_matrix()

        return self.img

    def _generate_feature_matrix(self):
        """Generates a feature matrix."""

        mask = None
        if self.training:
            # generate a randomized mask where 1 represents voxels used for training
            # the mask needs to be binary, where the value 1 is considered as a voxel which is to be loaded
            # we have following labels:
            # - 0 (background)
            # - 1 (white matter)
            # - 2 (grey matter)
            # - 3 (Hippocampus)
            # - 4 (Amygdala)
            # - 5 (Thalamus)

            # you can exclude background voxels from the training mask generation
            # mask_background = self.img.images[structure.BrainImageTypes.BrainMask]
            # and use background_mask=mask_background in get_mask()

            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                self.img.images[structure.BrainImageTypes.GroundTruth],
                [0, 1, 2, 3, 4, 5],
                [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02])

            # convert the mask to a logical array where value 1 is False and value 0 is True
            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # generate features
        data = np.concatenate(
            [self._image_as_numpy_array(image, mask) for id_, image in self.img.feature_images.items()],
            axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray = None):
        """Gets an image as numpy array where each row is a voxel and each column is a feature.

        Args:
            image (sitk.Image): The image.
            mask (np.ndarray): A mask defining which voxels to return. True is background, False is a masked voxel.

        Returns:
            np.ndarray: An array where each row is a voxel and each column is a feature.
        """

        number_of_components = image.GetNumberOfComponentsPerPixel()  # the number of features for this image
        no_voxels = np.prod(image.GetSize())
        image = sitk.GetArrayFromImage(image)

        if mask is not None:
            no_voxels = np.size(mask) - np.count_nonzero(mask)

            if number_of_components == 1:
                masked_image = np.ma.masked_array(image, mask=mask)
            else:
                # image is a vector image, make a vector mask
                vector_mask = np.expand_dims(mask, axis=3)  # shape is now (z, x, y, 1)
                vector_mask = np.repeat(vector_mask, number_of_components,
                                        axis=3)  # shape is now (z, x, y, number_of_components)
                masked_image = np.ma.masked_array(image, mask=vector_mask)

            image = masked_image[~masked_image.mask]

        return image.reshape((no_voxels, number_of_components))


def pre_process(id_: str, paths: dict, norm_method: str = 'no', artifact_method: str = 'none',
                standard_scales: tuple = None, percs: np.array = None, **kwargs) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.
        norm_method (str): Normalization method
        artifact_method (str): Artifact method
        standard_scales (tuple): Standard scaling for histogram matching normalization for T1w and T2w images
        percs (np.array): Percentiles for histogram matching

    Returns:
        (structure.BrainImage):
    """

    print('-' * 10, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    transform = sitk.ReadTransform(path_to_transform)
    img = structure.BrainImage(id_, path, img, transform)

    print('artifact method: ' + artifact_method)
    if kwargs.get('artifact_pre', False):
        add_artifact(img, artifact_method)


    # construct pipeline for brain mask registration
    # we need to perform this before the T1w and T2w pipeline because the registered mask is used for skull-stripping
    pipeline_brain_mask = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_brain_mask.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                                      len(pipeline_brain_mask.filters) - 1)

    # execute pipeline on the brain mask image
    img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
        img.images[structure.BrainImageTypes.BrainMask])

    # construct pipeline for T1w image pre-processing
    pipeline_t1 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t1.add_filter(fltr_prep.SkullStripping())
        pipeline_t1.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('normalization_pre', False):
        if norm_method is 'hm':
            pipeline_t1.add_filter(fltr_prep.ImageNormalization(img.id_, 'T1w', norm_method, standard_scales[0], percs,
                                                                mask=img.images[structure.BrainImageTypes.BrainMask]))

        else:
            pipeline_t1.add_filter(fltr_prep.ImageNormalization(img.id_, 'T1w', norm_method,
                                                                mask=img.images[structure.BrainImageTypes.BrainMask]))

    # execute pipeline on the T1w image
    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1w])

    # construct pipeline for T2w image pre-processing
    pipeline_t2 = fltr.FilterPipeline() #  artifacts lead to wrong registration. why??
    if kwargs.get('registration_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2.set_param(fltr_prep.ImageRegistrationParameters(atlas_t2, img.transformation),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t2.add_filter(fltr_prep.SkullStripping())
        pipeline_t2.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('normalization_pre', False):
        if norm_method is 'hm':
            pipeline_t2.add_filter(fltr_prep.ImageNormalization(img.id_, 'T2w', norm_method, standard_scales[1], percs,
                                                                mask=img.images[structure.BrainImageTypes.BrainMask]))

        else:
            pipeline_t2.add_filter(fltr_prep.ImageNormalization(img.id_, 'T2w', norm_method,
                                                                mask=img.images[structure.BrainImageTypes.BrainMask]))

    # execute pipeline on the T2w image
    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2w])

    # construct pipeline for ground truth image pre-processing
    pipeline_gt = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_gt.add_filter(fltr_prep.ImageRegistration())
        pipeline_gt.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                              len(pipeline_gt.filters) - 1)

    # execute pipeline on the ground truth image
    img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
        img.images[structure.BrainImageTypes.GroundTruth])

    # update image properties to atlas image properties after registration
    img.image_properties = conversion.ImageProperties(img.images[structure.BrainImageTypes.T1w])

    # extract the features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()

    img.feature_images = {}  # we free up memory because we only need the img.feature_matrix
    # for training of the classifier

    return img


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation.

    Args:
        img (structure.BrainImage): The image.
        segmentation (sitk.Image): The segmentation (label image).
        probability (sitk.Image): The probabilities images (a vector image).

    Returns:
        sitk.Image: The post-processed image.
    """

    print('-' * 10, 'Post-processing', img.id_)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())
    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1w],
                                                     img.images[structure.BrainImageTypes.T2w],
                                                     probability), len(pipeline.filters) - 1)

    return pipeline.execute(segmentation)


def init_evaluator(directory: str, result_file_name: str = 'results.csv') -> eval_.Evaluator:
    """Initializes an evaluator.

    Args:
        directory (str): The directory for the results file.
        result_file_name (str): The result file name (CSV file).

    Returns:
        eval.Evaluator: An evaluator.
    """
    os.makedirs(directory, exist_ok=True)  # generate result directory, if it does not exists

    evaluator = eval_.Evaluator(eval_.ConsoleEvaluatorWriter(5))
    evaluator.add_writer(eval_.CSVEvaluatorWriter(os.path.join(directory, result_file_name)))
    evaluator.add_label(1, 'WhiteMatter')
    evaluator.add_label(2, 'GreyMatter')
    evaluator.add_label(3, 'Hippocampus')
    evaluator.add_label(4, 'Amygdala')
    evaluator.add_label(5, 'Thalamus')
    evaluator.metrics = [metric.DiceCoefficient(), metric.HausdorffDistance()]
    # warnings.warn('Initialized evaluation with the Dice coefficient. Do you know other suitable metrics?')
    # you should add more metrics than just the Hausdorff distance!
    return evaluator


def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict = None, norm_method: str = 'no', artifact_method: str = 'none',
                      multi_process=True) -> t.List[structure.BrainImage]:
    """Loads and pre-processes a batch of images.

    The pre-processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        data_batch (Dict[structure.BrainImageTypes, structure.BrainImage]): Batch of images to be processed.
        pre_process_params (dict): Pre-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.
        norm_method (str): Normalization method
        artifact_method (str): Artifact method

    Returns:
        List[structure.BrainImage]: A list of images.
    """
    if pre_process_params is None:
        pre_process_params = {}

    params_list = list(data_batch.items())
    if multi_process:
        images = mproc.MultiProcessor.run(pre_process, params_list, pre_process_params, mproc.PreProcessingPickleHelper)
    else:
        if norm_method is 'hm':
            images_unprocessed = [hm_load_images(id_, path) for id_, path in params_list]
            standard_scales, percs = hist_to_match(images_unprocessed)
            images_unprocessed.clear()
            images = [pre_process(id_, path, norm_method=norm_method, standard_scales=standard_scales, percs=percs,
                                  **pre_process_params) for id_, path in params_list]
        else:
            images = [pre_process(id_, path, norm_method=norm_method, artifact_method=artifact_method,
                      **pre_process_params) for id_, path in params_list]

    return images


def post_process_batch(brain_images: t.List[structure.BrainImage], segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image], post_process_params: dict = None,
                       multi_process=True) -> t.List[sitk.Image]:
    """ Post-processes a batch of images.

    Args:
        brain_images (List[structure.BrainImageTypes]): Original images that were used for the prediction.
        segmentations (List[sitk.Image]): The predicted segmentation.
        probabilities (List[sitk.Image]): The prediction probabilities.
        post_process_params (dict): Post-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[sitk.Image]: List of post-processed images
    """
    if post_process_params is None:
        post_process_params = {}

    param_list = zip(brain_images, segmentations, probabilities)
    if multi_process:
        pp_images = mproc.MultiProcessor.run(post_process, param_list, post_process_params,
                                             mproc.PostProcessingPickleHelper)
    else:
        pp_images = [post_process(img, seg, prob, **post_process_params) for img, seg, prob in param_list]
    return pp_images

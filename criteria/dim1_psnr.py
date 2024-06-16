import tensorflow as tf
def dim1_psnr(y_true, y_pred):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) for 1D (grayscale) images.

    Args:
        y_true (tf.Tensor): The true/original 1D image.
        y_pred (tf.Tensor): The predicted/reconstructed 1D image.

    Returns:
        tf.Tensor: The PSNR value.
    """
    max_pixel = 255.0  # Assuming 8-bit pixel values (0-255)
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
    psnr = 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr
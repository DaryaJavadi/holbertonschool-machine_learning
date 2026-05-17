def generate_image(self, iterations=1000, step=None, lr=0.01,
                   beta1=0.9, beta2=0.99):
    """
    This method generates the Neural Style Transfer image.
    Args:
    iterations (int): the number of iterations to perform gradient
        descent over.
    step (int): the step at which to print information about the
        training, including the final iteration:
        print: Cost at iteration {i}: {J_total}, content
        {J_content}, style {J_style}
            i is the iteration
            J_total is the total cost
            J_content is the content cost
            J_style is the style cost
    lr (float): the learning rate for gradient descent.
    beta1 (float): the beta1 parameter for gradient descent.
    beta2 (float): the beta2 parameter for gradient descent.
    Returns:
    (generated_image, cost) where:
        generated_image is the generated image.
        cost is the cost of the generated image.
    """
    if not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    if step is not None and not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step is not None and (step <= 0 or step >= iterations):
        raise ValueError("step must be positive and less than iterations")

    if not isinstance(lr, (float, int)):
        raise TypeError("lr must be a number")
    if lr <= 0:
        raise ValueError("lr must be positive")

    if not isinstance(beta1, float):
        raise TypeError("beta1 must be a float")
    if beta1 < 0 or beta1 > 1:
        raise ValueError("beta1 must be in the range [0, 1]")

    if not isinstance(beta2, float):
        raise TypeError("beta2 must be a float")
    if beta2 < 0 or beta2 > 1:
        raise ValueError("beta2 must be in the range [0, 1]")

    # Initialize optimizer
    optimizer = tf.optimizers.Adam(
        learning_rate=lr,
        beta_1=beta1,
        beta_2=beta2
    )

    # Initialize generated image
    generated_image = tf.Variable(self.content_image)

    # Track best image and cost
    best_cost = float('inf')
    best_image = None

    # Optimization loop
    for i in range(iterations):

        grads, J_total, J_content, J_style, J_var = \
            self.compute_grads(generated_image)

        optimizer.apply_gradients([(grads, generated_image)])

        # Clip values to valid image range
        generated_image.assign(
            tf.clip_by_value(generated_image, 0, 1)
        )

        # Print progress
        if step is not None and (
                i % step == 0 or i == iterations - 1):
            print(
                f"Cost at iteration {i}: {J_total.numpy()}, "
                f"content {J_content.numpy()}, "
                f"style {J_style.numpy()}, "
                f"var {J_var.numpy()}"
            )

        # Save best result
        if J_total < best_cost:
            best_cost = J_total
            best_image = tf.identity(generated_image)

    # Remove batch dimension
    best_image = best_image[0]

    return best_image.numpy(), best_cost.numpy()
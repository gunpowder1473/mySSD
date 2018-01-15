import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from collections import namedtuple

slim = tf.contrib.slim

Clone = namedtuple('Clone', ['outputs', 'scope', 'device', ])


@add_arg_scope
def l2_normalization(inputs, scaling=False, scale_initializer=tf.ones_initializer(), reuse=None,
                     variables_collections=None, outputs_collections=None,
                     data_format='NHWC', trainable=True, scope=None):
    with tf.variable_scope(scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            norm_dim = tf.range(inputs_rank - 1, inputs_rank)
            params_shape = inputs_shape[-1:]
        elif data_format == 'NCHW':
            norm_dim = tf.range(1, 2)
            params_shape = (inputs_shape[1])

        outputs = tf.nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        if scaling:
            scale_collections = utils.get_variable_collections(variables_collections, 'scale')
            scale = variables.model_variable('gamma', shape=params_shape, dtype=dtype, initializer=scale_initializer,
                                             collections=scale_collections, trainable=trainable)
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs, scale)

        return utils.collect_named_outputs(outputs_collections, sc.original_name_scope, outputs)


@add_arg_scope
def dataFormatChange(inputs, data_format='NHWC', scope=None):
    with tf.name_scope(scope, 'data_format_change', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net


def tensorShape(tensor, rank=3):
    if tensor.get_shape().is_fully_defined():
        return tensor.get_shape().as_list()
    else:
        static_shape = tensor.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = tensor.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(tensor), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def listReshape(l, shape=None):
    r = []
    if shape is None:
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i + s])
            i += s
    return r


def creatClones(config, model_fn, args=None, kwargs=None):
    clones = []
    args = args or []
    kwargs = kwargs or {}
    # 为slim.model_variable, slim.variable加默认参数
    with slim.arg_scope([slim.model_variable, slim.variable], device=config.variables_device()):
        for i in range(0, config.num_clones):
            with tf.name_scope(config.clone_scope(i)) as clone_scope:
                clone_device = config.clone_device(i)
                with tf.device(clone_device):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
                        outputs = model_fn(*args, **kwargs)
                    clones.append(Clone(outputs, clone_scope, clone_device))
    return clones


def setLearningRate(flags, num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / flags.batch_size * flags.num_epochs_per_decay)

    if flags.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(flags.learning_rate,
                                          global_step,
                                          decay_steps,
                                          flags.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif flags.learning_rate_decay_type == 'fixed':
        return tf.constant(flags.learning_rate, name='fixed_learning_rate')
    elif flags.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(flags.learning_rate,
                                         global_step,
                                         decay_steps,
                                         flags.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         flags.learning_rate_decay_type)


def setOptimizer(flags, learning_rate):
    if flags.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=flags.adadelta_rho,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=flags.adagrad_initial_accumulator_value)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=flags.adam_beta1,
            beta2=flags.adam_beta2,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=flags.ftrl_learning_rate_power,
            initial_accumulator_value=flags.ftrl_initial_accumulator_value,
            l1_regularization_strength=flags.ftrl_l1,
            l2_regularization_strength=flags.ftrl_l2)
    elif flags.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=flags.momentum,
            name='Momentum')
    elif flags.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=flags.rmsprop_decay,
            momentum=flags.rmsprop_momentum,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
    return optimizer


def getTrainableVariables(flags):
    if flags.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def optimizerClones(clones, optimizer, regularization_losses=None, **kwargs):
    grads_and_vars = []
    clones_losses = []
    num_clones = len(clones)
    if regularization_losses is None:
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for clone in clones:
        with tf.name_scope(clone.scope):
            clone_loss, clone_grad = _optimize_clone(optimizer, clone, num_clones, regularization_losses, **kwargs)
            if clone_loss is not None:
                clones_losses.append(clone_loss)
                grads_and_vars.append(clone_grad)
            regularization_losses = None
    total_loss = tf.add_n(clones_losses, name='total_loss')
    grads_and_vars = _sum_clones_gradients(grads_and_vars)
    return total_loss, grads_and_vars


def setInit(flags):
    if flags.checkpoint_path is None:
        return None
    if tf.train.latest_checkpoint(flags.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % flags.train_dir)
        return None
    exclusions = []
    if flags.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in flags.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    if flags.checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(flags.model_name,
                                 flags.checkpoint_model_scope): var
             for var in variables_to_restore}

    if tf.gfile.IsDirectory(flags.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path)
    else:
        checkpoint_path = flags.checkpoint_path
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=flags.ignore_missing_vars)


@add_arg_scope
def pad2d(inputs, pad=(0, 0), mode='CONSTANT', data_format='NHWC', trainable=True, scope=None):
    with tf.name_scope(scope, 'pad2d', [inputs]):
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net


def abs_smooth(x):
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
                    **kwargs):
    """Compute losses and gradients for a single clone.

    Args:
        optimizer: A tf.Optimizer  object.
        clone: A Clone namedtuple.
        num_clones: The number of clones being deployed.
        regularization_losses: Possibly empty list of regularization_losses
            to add to the clone losses.
        **kwargs: Dict of kwarg to pass to compute_gradients().

    Returns:
        A tuple (clone_loss, clone_grads_and_vars).
            - clone_loss: A tensor for the total loss for the clone.  Can be None.
            - clone_grads_and_vars: List of (gradient, variable) for the clone.
                Can be empty.
    """
    sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
    clone_grad = None
    if sum_loss is not None:
        with tf.device(clone.device):
            clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return sum_loss, clone_grad


def _gather_clone_loss(clone, num_clones, regularization_losses):
    """Gather the loss for a single clone.

    Args:
        clone: A Clone namedtuple.
        num_clones: The number of clones being deployed.
        regularization_losses: Possibly empty list of regularization_losses
            to add to the clone losses.

    Returns:
        A tensor for the total loss for the clone.  Can be None.
    """
    # The return value.
    sum_loss = None
    # Individual components of the loss that will need summaries.
    clone_loss = None
    regularization_loss = None
    # Compute and aggregate losses on the clone device.
    with tf.device(clone.device):
        all_losses = []
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        if clone_losses:
            clone_loss = tf.add_n(clone_losses, name='clone_loss')
            if num_clones > 1:
                clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                                    name='scaled_clone_loss')
            all_losses.append(clone_loss)
        if regularization_losses:
            regularization_loss = tf.add_n(regularization_losses,
                                           name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
    # Add the summaries out of the clone device block.
    if clone_loss is not None:
        tf.summary.scalar('clone_loss', clone_loss)
        # tf.summary.scalar(clone.scope + '/clone_loss', clone_loss)
    if regularization_loss is not None:
        tf.summary.scalar('regularization_loss', regularization_loss)
    return sum_loss


def _sum_clones_gradients(clone_grads):
    """Calculate the sum gradient for each shared variable across all clones.

    This function assumes that the clone_grads has been scaled appropriately by
    1 / num_clones.

    Args:
      clone_grads: A List of List of tuples (gradient, variable), one list per
        `Clone`.

    Returns:
      List of tuples of (gradient, variable) where the gradient has been summed
        across all clones.
    """
    sum_grads = []
    for grad_and_vars in zip(*clone_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(g)
        if grads:
            if len(grads) > 1:
                sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))
    return sum_grads

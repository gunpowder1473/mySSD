import tensorflow as tf


class DeploymentConfig(object):
    def __init__(self,
                 num_clones=1,
                 clone_on_cpu=False,
                 fake_multiple_gpus=False,
                 replica_id=0,
                 num_replicas=1,
                 num_ps_tasks=0,
                 worker_job_name='worker',
                 ps_job_name='ps'):
        if num_replicas > 1:
            if num_ps_tasks < 1:
                raise ValueError('When using replicas num_ps_tasks must be positive')
        if num_replicas > 1 or num_ps_tasks > 0:
            if not worker_job_name:
                raise ValueError('Must specify worker_job_name when using replicas')
            if not ps_job_name:
                raise ValueError('Must specify ps_job_name when using parameter server')
        if replica_id >= num_replicas:
            raise ValueError('replica_id must be less than num_replicas')
        self._num_clones = num_clones
        self._clone_on_cpu = clone_on_cpu
        self._fake_multiple_gpus = fake_multiple_gpus
        self._replica_id = replica_id
        self._num_replicas = num_replicas
        self._num_ps_tasks = num_ps_tasks
        self._ps_device = '/job:' + ps_job_name if num_ps_tasks > 0 else ''
        self._worker_device = '/job:' + worker_job_name if num_ps_tasks > 0 else ''

    @property
    def num_clones(self):
        return self._num_clones

    @property
    def clone_on_cpu(self):
        return self._clone_on_cpu

    @property
    def fake_multiple_gpus(self):
        return self._fake_multiple_gpus

    @property
    def replica_id(self):
        return self._replica_id

    @property
    def num_replicas(self):
        return self._num_replicas

    @property
    def num_ps_tasks(self):
        return self._num_ps_tasks

    @property
    def ps_device(self):
        return self._ps_device

    @property
    def worker_device(self):
        return self._worker_device

    def caching_device(self):
        if self._num_ps_tasks > 0:
            return lambda op: op.device
        else:
            return None

    def clone_device(self, clone_index):
        if clone_index >= self._num_clones:
            raise ValueError('clone_index must be less than num_clones')
        device = ''
        if self._num_ps_tasks > 0:
            device += self._worker_device
        if self._clone_on_cpu:
            device += '/device:CPU:0'
        else:
            if self._num_clones > 1 and not self._fake_multiple_gpus:
                device += '/device:GPU:%d' % clone_index
        return device

    def clone_scope(self, clone_index):
        if clone_index >= self._num_clones:
            raise ValueError('clone_index must be less than num_clones')
        scope = ''
        if self._num_clones > 1:
            scope = 'clone_%d' % clone_index
        return scope

    def optimizer_device(self):
        if self._num_ps_tasks > 0 or self._num_clones > 0:
            return self._worker_device + '/device:CPU:0'
        else:
            return ''

    def inputs_device(self):
        device = ''
        if self._num_ps_tasks > 0:
            device += self._worker_device
        device += '/device:CPU:0'
        return device

    def variables_device(self):
        device = ''
        if self._num_ps_tasks > 0:
            device += self._ps_device
        device += '/device:CPU:0'

        class _PSDeviceChooser(object):
            def __init__(self, device, tasks):
                self._device = device
                self._tasks = tasks
                self._task = 0

            def choose(self, op):
                if op.device:
                    return op.device
                node_def = op if isinstance(op, tf.NodeDef) else op.node_def
                if node_def.op == 'Variable':
                    t = self._task
                    self._task = (self._task + 1) % self._tasks
                    d = '%s/task:%d' % (self._device, t)
                    return d
                else:
                    return op.device

        if not self._num_ps_tasks:
            return device
        else:
            chooser = _PSDeviceChooser(device, self._num_ps_tasks)
            return chooser.choose

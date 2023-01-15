import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec


class MMoE(Layer):

    def __init__(self,
                 units,  # 隐藏单元的数量
                 num_experts,  # 专家数量
                 num_tasks,   # 任务数量
                 use_expert_bias=True,  # 布尔值，表示专家权重中偏差的使用
                 use_gate_bias=True,   # 布尔值，表示门控权重中偏置的使用
                 expert_activation='relu',  # 专家权重的激活函数
                 gate_activation='softmax',  # 门权重的激活函数
                 expert_bias_initializer='zeros',  # 专家偏差的初始化器
                 gate_bias_initializer='zeros',  # 门偏差的初始化器
                 expert_bias_regularizer=None,  # 专家偏差的正则化
                 gate_bias_regularizer=None,  # 门偏差的正则化
                 expert_bias_constraint=None,  # 专家偏差的约束
                 gate_bias_constraint=None,  # 门偏差的约束
                 expert_kernel_initializer='VarianceScaling',  # 专家权重的初始化式
                 gate_kernel_initializer='VarianceScaling',   # 门权重的初始化式
                 expert_kernel_regularizer=None,  # 专家权重的正则化
                 gate_kernel_regularizer=None,  # 门权重的正则化
                 expert_kernel_constraint=None,  # 专家权值的约束
                 gate_kernel_constraint=None,  # 门权值的约束
                 activity_regularizer=None,  # activity的正则化器
                 **kwargs):
        # 隐藏节点参数
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # 权重参数
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # 激活参数
        self.expert_activation = activations.get(expert_activation)
        self.gate_activation = activations.get(gate_activation)

        # 偏差参数
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity参数
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Keras参数
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        super(MMoE, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: Keras张量(未来层的输入)或Keras张量的列表/元组，以参考权重形状计算
        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = input_shape[-1]

        # 初始化专家权重   输入特征数量*每个专家单元数量*专家数量，为shape
        self.expert_kernels = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.units, self.num_experts),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
        )

        # 初始化专家偏置     每个专家单元数量*专家个数，为shape
        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name='expert_bias',
                shape=(self.units, self.num_experts),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint,
            )

        # 初始化门权重    输入特征数量*专家的数量*任务的数量，为shape
        # 门与任务的个数有关
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_experts),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        ) for i in range(self.num_tasks)]

        # 初始化门偏置 (number of experts * number of tasks)
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias_task_{}'.format(i),
                shape=(self.num_experts,),
                initializer=self.gate_bias_initializer,
                regularizer=self.gate_bias_regularizer,
                constraint=self.gate_bias_constraint
            ) for i in range(self.num_tasks)]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dimension})

        super(MMoE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # 计算MMOE输出的主要函数
        gate_outputs = []
        final_outputs = []

        # f_{i}(x) = activation(W_{i} * x + b)
        expert_outputs = tf.tensordot(a=inputs, b=self.expert_kernels, axes=1)

        if self.use_expert_bias:
            expert_outputs = K.bias_add(x=expert_outputs, bias=self.expert_bias)
        expert_outputs = self.expert_activation(expert_outputs)

        # g^{k}(x) = activation(W_{gk} * x + b)
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = K.dot(x=inputs, y=gate_kernel)

            if self.use_gate_bias:
                gate_output = K.bias_add(x=gate_output, bias=self.gate_bias[index])
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            expanded_gate_output = K.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(K.sum(weighted_expert_output, axis=2))

        return final_outputs

    def compute_output_shape(self, input_shape):
        # 计算MMoE层输出形状的方法
        assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

    def get_config(self):
        #  返回MMoE层配置的方法
        config = {
            'units': self.units,
            'num_experts': self.num_experts,
            'num_tasks': self.num_tasks,
            'use_expert_bias': self.use_expert_bias,
            'use_gate_bias': self.use_gate_bias,
            'expert_activation': activations.serialize(self.expert_activation),
            'gate_activation': activations.serialize(self.gate_activation),
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gate_bias_initializer': initializers.serialize(self.gate_bias_initializer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gate_bias_regularizer': regularizers.serialize(self.gate_bias_regularizer),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gate_bias_constraint': constraints.serialize(self.gate_bias_constraint),
            'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
            'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(MMoE, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

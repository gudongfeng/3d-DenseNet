"""3D DenseNet Model file"""
import tensorflow as tf


class DenseNet3D(object):
    """3D DenseNet Model class"""

    def __init__(
            self,
            video_clips,  # Shape: [batch_size, sequence_length, height, width, channels]
            labels,  # Shape: [batch_size, num_classes] 
            initial_learning_rate,
            decay_step,
            lr_decay_factor,
            num_classes,
            growth_rate,
            network_depth,
            total_blocks,
            keep_prob,
            weight_decay,
            reduction,
            bc_mode=False,
            **kwargs):
        self.video_clips = video_clips
        self.labels = labels
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.network_depth = network_depth

        # How many features will be received after first convolution value
        self.first_output_features = growth_rate * 2

        self.total_blocks = total_blocks
        self.layers_per_block = (network_depth -
                                 (total_blocks + 1)) // total_blocks

        # Compression rate at the transition layers
        self.reduction = reduction
        self.bc_mode = bc_mode
        if not bc_mode:
            self.reduction = 1.0
            print(
                "Build 3D DenseNet model with %d blocks, %d composite layers each."
                % (total_blocks, self.layers_per_block))

        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print(
                "Build 3D DenseNet-BC model with %d blocks, %d bottleneck layers and %d composite layers each."
                % (total_blocks, self.layers_per_block, self.layers_per_block))

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self._is_training = tf.convert_to_tensor(True)

        # Initialize the global step
        self.global_step = tf.train.get_or_create_global_step()

        self.learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            self.global_step,
            decay_step,
            lr_decay_factor,
            staircase=True)

        self._build_graph()

    def _build_graph(self):
        # First convolution layer
        with tf.variable_scope('Initial_convolution'):
            output = self._conv3d(
                self.video_clips,
                out_features_count=self.first_output_features,
                kernel_size=7,
                strides=[1, 1, 2, 2, 1])
            output = self._pool(output, k=3, d=2, k_stride=2, d_stride=1)

        # Add 3D DenseNet blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self._add_block(output, self.growth_rate,
                                         self.layers_per_block)
            # The last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self._transition_layer(output, pool_depth=2)

        # Fully connected layers
        with tf.variable_scope('Transition_to_classes'):
            self._logits = self._trainsition_layer_to_classes(output)

        # Prediction result
        self._prediction = tf.argmax(self._logits, 1)

        # Losses
        self._cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logits, labels=self.labels),
            name='Cross_entropy')
        self.l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.total_loss = self._cross_entropy + self.l2_loss * self.weight_decay

        # Optimizer and training op
        self._train_op = tf.contrib.layers.optimize_loss(
            loss=self.total_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer='Momentum')

    @property
    def logits(self):
        return self._logits

    @property
    def train_op(self):
        return self._train_op

    @property
    def losses(self):
        return self._cross_entropy

    @property
    def prediction(self):
        return self._prediction

    @property
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self._logits, 1), self.labels)
        return tf.metrics.mean(tf.cast(correct_prediction, tf.float32))

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, value):
        self._is_training = tf.convert_to_tensor(value)

    def _conv3d(self,
                inputs,
                out_features_count,
                kernel_size,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'):
        input_features_count = int(inputs.get_shape()[-1])
        kernel = tf.get_variable(
            'kernel',
            shape=[
                kernel_size, kernel_size, kernel_size, input_features_count,
                out_features_count
            ],
            initializer=tf.random_normal_initializer())
        with tf.name_scope('3d_conv'):
            return tf.nn.conv3d(
                inputs, filter=kernel, strides=strides, padding=padding)

    def _pool(self,
              inputs,
              k,
              d=2,
              k_stride=None,
              d_stride=None,
              width_k=None,
              k_stride_width=None):
        if not width_k:
            width_k = k
        kernel_size = [1, d, k, width_k, 1]
        if not k_stride:
            k_stride = k
        if not k_stride_width:
            k_stride_width = k_stride
        if not d_stride:
            d_stride = d
        strides = [1, d_stride, k_stride, k_stride_width, 1]
        return tf.nn.max_pool3d(
            inputs, ksize=kernel_size, strides=strides, padding='SAME')

    def _add_block(self, inputs, growth_rate, layers_per_block):
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                return self._add_internal_layer(inputs, growth_rate)

    def _add_internal_layer(self, inputs, growth_rate):
        if not self.bc_mode:
            composite_out = self._composite_function(
                inputs, out_features_count=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self._bottleneck(
                inputs, out_features_count=growth_rate)
            composite_out = self._composite_function(
                bottleneck_out, out_features_count=growth_rate, kernel_size=3)

        with tf.name_scope('concat'):
            return tf.concat(axis=4, values=(inputs, composite_out))

    def _composite_function(self, inputs, out_features_count, kernel_size):
        with tf.variable_scope('composite_function'):
            # Batch normalization
            output = self._batch_norm(inputs)
            # ReLU
            with tf.name_scope('ReLU'):
                output = tf.nn.relu(output)
            # Convolution
            output = self._conv3d(
                output,
                out_features_count=out_features_count,
                kernel_size=kernel_size)
            # Dropout
            output = self._dropout(output)
        return output

    def _bottleneck(self, inputs, out_features_count):
        with tf.variable_scope('bottleneck'):
            # Batch normalization
            output = self._batch_norm(inputs)
            # ReLU
            with tf.name_scope('ReLU'):
                output = tf.nn.relu(output)

            inter_features = out_features_count * 4
            output = self._conv3d(
                output,
                out_features_count=inter_features,
                kernel_size=1,
                padding='VALID')
            output = self._dropout(output)
        return output

    def _batch_norm(self, inputs):
        with tf.name_scope('batch_normalization'):
            output = tf.contrib.layers.batch_norm(
                inputs,
                scale=True,
                is_training=self._is_training,
                updates_collections=None)
        return output

    def _dropout(self, inputs):
        if self.keep_prob < 1:
            with tf.name_scope('dropout'):
                output = tf.cond(self._is_training,
                                 lambda: tf.nn.dropout(inputs, self.keep_prob),
                                 lambda: inputs)
        else:
            output = inputs
        return output

    def _transition_layer(self, inputs, pool_depth):
        out_features_count = int(int(inputs.get_shape()[-1]) * self.reduction)
        output = self._composite_function(
            inputs, out_features_count=out_features_count, kernel_size=1)
        with tf.name_scope('pooling'):
            output = self._pool(output, k=2, d=pool_depth)
        return output

    def _trainsition_layer_to_classes(self, inputs):
        # Batch normalization
        output = self._batch_norm(inputs)
        # ReLU
        with tf.name_scope('ReLU'):
            output = tf.nn.relu(output)
        # pooling
        last_pool_kernel_width = int(output.get_shape()[-2])
        last_pool_kernel_height = int(output.get_shape()[-3])
        last_sequence_length = int(output.get_shape()[1])
        with tf.name_scope('pooling'):
            output = self._pool(
                output,
                k=last_pool_kernel_height,
                d=last_sequence_length,
                width_k=last_pool_kernel_width,
                k_stride_width=last_pool_kernel_width)
        # Fully connected
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        weight = tf.get_variable(
            'fc_w',
            shape=[features_total, self.num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(
            'fc_bias',
            shape=[self.num_classes],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(output, weight) + bias
        return logits

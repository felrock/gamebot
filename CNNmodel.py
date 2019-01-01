import tensorflow as tf

class CNNModel(object):

    # static x and y?
    #x = tf.placeholder('float', [None, 784])
    #y = tf.placeholder('float')


    def __init__(self,
                 classnumber=10,
                 batchsize=128,
                 imagefolder='images/',
                 imagedimension=(200,200)):

        self.classnumber    = classnumber
        self.batchsize      = batchsize
        self.imagefolder    = imagefolder
        self.imagedimension = imagedimension

    @staticmethod
    def conv2d(x, w):

        return tf.nn.conv2d(x,
                            w,
                            strides=[1,1,1,1],
                            padding='SAME')
    @staticmethod
    def maxpool2d(x):

        return tf.nn.max_pool(x,
                              ksize=[1,1,1,1],
                              strides=[1,1,1,1],
                              padding='SAME')

    def loadLabels(self):
        #TODO:load labels from .txt file
        return None

    def makeBottlenecks(self):
        #make bottlenecks from images,
        return None

    def loadBatch(self):
        #TODO:load batch from bottlenecks
        return None

    def loadTestset(self):
        #TODO: load testset from bottlenecks
        # Could be executed when loading batches or
        # when creating the bottlenecks
        return None

    def convolutional_neural_network_model(self, x):

        weights = {'w_conv1':tf.Variable(tf.random_normal([5, 5, 1, 64])),
                   'w_conv2':tf.Variable(tf.random_normal([5, 5, 32, 128])),
                   'w_fc':tf.Variable(tf.random_normal([7*7*64, 2048])),
                   'out':tf.Variable(tf.random_normal([2048, self.classnumber]))}

        biases = {'b_conv1':tf.Variable(tf.random_normal([64])),
                  'b_conv2':tf.Variable(tf.random_normal([128])),
                  'b_fc':tf.Variable(tf.random_normal([2048])),
                  'out':tf.Variable(tf.random_normal([self.classnumber]))}

        x = tf.reshape(x, shape=[-1,self.imagedimension[0],self.imagedimension[0],1])
        conv1 = CNNModel.conv2d(x, weights['w_conv1'])
        conv1 = CNNModel.maxpool2d(conv1)

        conv2 = CNNModel.conv2d(conv1, weights['w_conv2'])
        conv2 = CNNModel.maxpool2d(conv2)

        fc = tf.reshape(conv2,[-1, 7*7*64])
        fc = tf.nn.relu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])

        output = tf.matmul(fc, weights['out']) + biases['out']
        return output

    def train_neural_network(self):
        imageflat = self.imagedimension[0]*self.imagedimension[1]
        x = tf.placeholder('float', [None, imageflat])
        y = loadLabels()
        prediction = self.convolutional_neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                      labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epochs = 10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(self.imagesetsize/self.batchsize):
                    epoch_x, epoch_y = loadBatch()
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, '/', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            testset = loadTestset()
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval(testset))

            # TODO: Add support for saving the tensorflow model
            tf.saved_model.simple_save(
                sess, path, inputs_dict, outputs_dict
            )

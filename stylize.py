import vgg

import tensorflow as tf
import numpy as np
from neural_style import *
from sys import stderr

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
STYLE_LAYERS = ('relu1_1',)
STYLE_LAYERS = ('relu1_1', 'relu2_1',)


def stylize(network, initial, content, styles, iterations,
        content_weight, style_weight, style_blend_weights, tv_weight,
        learning_rate, print_iterations=None, checkpoint_iterations=None):
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(network, image)
        content_pre = np.array([vgg.preprocess(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net, _ = vgg.net(network, image)
            style_pre = np.array([vgg.preprocess(styles[i], mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                print 'Initial feature shape: ', features.shape
                features = np.reshape(features, (-1, features.shape[3]))
                #mask = np.zeros_like(features)
                #mask[:49664/2, :] = 1
                #print 'Mask shape', mask.shape
                print 'Final features shape', features.shape
                #features = features*mask
                gram = np.matmul(features.T, features) / features.size
                print 'Gram matrix shape: ', gram.shape
                style_features[i][layer] = gram

    #sys.exit()
    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, mean_pixel)])
            initial = initial.astype('float32')
        image = tf.Variable(initial)
        net, _ = vgg.net(network, image)

        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                content_features[CONTENT_LAYER].size)
        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                print 'Height, width, number', height, width, number
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                
                #print tf.shape(feats).as_list()
                print 'Height', height
                print 'Weight', width
                print 'Number', number
                print 'Style features shape', style_features[i][style_layer].shape
                print style_layer
                
                if style_layer == 'relu2_1':
                    mask = np.zeros((height*width, number), dtype=np.float32)
                    temp = imread('emma/emma_test_mask.jpg').astype(np.float32)
                    c = temp.reshape(height,2,width,2)
                    temp = c.max(axis=1).max(axis=2)
                    print temp.shape
                    maskt = np.reshape(temp, (height*width,))
                    maskt = maskt > 100
                    for d in xrange(number):
                        mask[:,d] = maskt
                    print 'Mask shape', mask.shape
                    #b = mask.reshape(height*width*2, 2, number/2,2)
                    #mask = b.max(axis=1).max(axis=2)
                    #print 'New mask shape', mask.shape
                else:
                    mask = np.zeros((height*width, number), dtype=np.float32)
                    maskt = np.reshape(imread('emma/emma_test_mask.jpg').astype(np.float32), (height*width,))
                    maskt = maskt > 100
                    for d in xrange(number):
                        mask[:,d] = maskt
                    print 'Mask shape', mask.shape
                if i == 0:
                    mask = tf.constant(mask)
                    print 'Mask shape', map(lambda i: i.value, mask.get_shape())
                    feats = tf.mul(feats,mask)

                    gram = tf.matmul(tf.transpose(feats), feats) / size
                    style_gram = style_features[i][style_layer]
                    style_losses.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
                else:
                    mask2 = mask < 1
                    feats2 = tf.mul(feats,mask2)
                    gram2 = tf.matmul(tf.transpose(feats2), feats2) / size
                    style_gram = style_features[i][style_layer]
                    style_losses.append(2 * tf.nn.l2_loss(gram2 - style_gram) / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        def print_progress(i, last=False):
            if print_iterations is not None:
                if i is not None and i % print_iterations == 0 or last:
                    print >> stderr, '  content loss: %g' % content_loss.eval()
                    print >> stderr, '    style loss: %g' % style_loss.eval()
                    print >> stderr, '       tv loss: %g' % tv_loss.eval()
                    print >> stderr, '    total loss: %g' % loss.eval()

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(iterations):
                print_progress(i)
                print >> stderr, 'Iteration %d/%d' % (i + 1, iterations)
                train_step.run()
                if (checkpoint_iterations is not None and
                        i % checkpoint_iterations == 0) or i == iterations - 1:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()
                print_progress(None, i == iterations - 1)

                if i % 10 == 0 and best is not None:
                    tmp_img = vgg.unprocess(best.reshape(shape[1:]), mean_pixel)
                    imsave("iter" + str(i) + ".jpg", tmp_img)

            return vgg.unprocess(best.reshape(shape[1:]), mean_pixel)


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

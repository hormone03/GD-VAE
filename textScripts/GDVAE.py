from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
import os
import utils as utils
import sys
import argparse
import pickle
from absl import app

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

np.random.seed(0)
tf.set_random_seed(0)

# flags = tf.app.flags
flags = app.flags
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_integer('n_hidden', 100, 'Size of each hidden layer.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity of the MLP.')
flags.DEFINE_string('summaries_dir', 'summaries', 'where to save the summaries')
FLAGS = flags.FLAGS


class  DVAE(object):

    def __init__(self,
                 analytical,
                 vocab_size,
                 n_hidden,
                 n_topic,
                 n_sample,
                 learning_rate,
                 batch_size,
                 non_linearity,
                 adam_beta1,
                 adam_beta2,
                 B,
                 gd_prior_alpha,
                 gd_prior_beta,
                 correction):
        tf.compat.v1.reset_default_graph()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, vocab_size],
                                name='input')  # dataType, shape : None means any no of samples, name
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.warm_up = tf.placeholder(tf.float32, (), name='warm_up')  # warm up
        self.B = tf.placeholder(tf.int32, (), name='B')
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2  # adam_beta is reference to adam_beta defined in parseArgs; nso no need of placeholder
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.min_alpha = tf.placeholder(tf.float32, (), name='min_alpha')
        self.min_beta = tf.placeholder(tf.float32, (), name='min_beta')

        # encoder
        with tf.variable_scope('encoder'):
            self.enc_vec = utils.mlp(self.x, [self.n_hidden],
                                     self.non_linearity)  # using Multilayer perceptron; the fully connected linear layer
            self.enc_vec = tf.nn.dropout(self.enc_vec, self.keep_prob)
            self.mean = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean'))
            # Me: Take minimum between min_alpha and softplus(x)=log(1+e^x)
            # Me: Take minimum between min_beta and softplus(x)=log(1+e^x)
            self.alpha = tf.maximum(self.min_alpha, tf.log(1. + tf.exp(self.mean)))
            self.beta_ = tf.nn.elu(self.mean)
            self.beta = tf.maximum(self.min_beta, tf.log(1. + tf.exp(self.beta_)))

            # Dirichlet prior alpha0
            # me: Dirichlet prior alpha0; len(input)=self.batch_size
            self.prior_alpha = tf.ones((batch_size, self.n_topic), dtype=tf.float32,
                                       name='prior_alpha') * gd_prior_alpha
            self.prior_beta = tf.ones((batch_size, self.n_topic), dtype=tf.float32,
                                      name='prior_beta') * gd_prior_beta

            decoderParamSum = self.alpha + self.beta
            priorParamSum = self.prior_alpha + self.prior_beta
            alphaParamsDiff = self.alpha - self.prior_alpha
            numerator = tf.lgamma(decoderParamSum) + tf.lgamma(self.prior_alpha) + tf.lgamma(self.prior_beta)
            denomirator = tf.lgamma(self.alpha) + tf.lgamma(self.beta) + tf.lgamma(priorParamSum)
            firstTerm = tf.reduce_sum((numerator - denomirator), axis=1)
            secondTerm = tf.reduce_sum((tf.digamma(decoderParamSum) - tf.digamma(self.beta)), axis=1)
            secondTerm = tf.reshape(secondTerm, (batch_size, 1))
            secondTerm = tf.digamma(self.alpha) - tf.digamma(self.beta) - secondTerm
            secondTerm = tf.reduce_sum(tf.multiply(alphaParamsDiff, secondTerm), axis=1)
            thirdTerm = tf.reduce_sum((tf.digamma(decoderParamSum) - tf.digamma(self.beta)), axis=1)
            thirdTerm = tf.reshape(thirdTerm, (batch_size, 1))
            v1 = tf.concat([self.beta[:, :-1] - self.alpha[:, 1:] - self.beta[:, 1:], self.beta[:, -1:] - 1], axis=-1)
            v2 = tf.concat([self.prior_beta[:, :-1] - self.prior_alpha[:, 1:] - self.prior_beta[:, 1:],
                            self.prior_beta[:, -1:] - 1], axis=-1)
            thirdTerm = tf.reduce_sum((tf.multiply((v1 - v2), thirdTerm)), axis=1)
            self.analytical_kld = firstTerm - secondTerm + thirdTerm
            self.analytical_kld = self.mask * self.analytical_kld  # mask paddings

        with tf.variable_scope('decoder'):
            # NB: I'm using only one sample, so, no need to consider multiple samples
            # sample gammas
            # Me: Z_tilde, section 3.4
            gam_alpha = tf.squeeze(tf.random_gamma(shape=(1,), alpha=self.alpha + tf.to_float(self.B)))
            gam_beta = tf.squeeze(tf.random_gamma(shape=(1,), alpha=self.beta + tf.to_float(self.B)))
            # Z
            z_add = gam_alpha + gam_beta
            gam_alpha_beta = tf.div(gam_alpha, z_add)

            # reverse engineer the random variables used in the gamma rejection sampler
            # Me: calc_epsilon cal episelon, eq2: make episelon subj of fomular.. h(epselon, alpha) is Z_tilde
            # Me: stop_gradient -> .detach() in pytorch i.e with no gradient
            eps_alpha = tf.stop_gradient(calc_epsilon(gam_alpha, self.alpha + tf.to_float(self.B)))
            eps_beta = tf.stop_gradient(calc_epsilon(gam_beta, self.beta + tf.to_float(self.B)))
            # uniform variables for shape augmentation of gamma
            u = tf.random_uniform((self.B, batch_size, self.n_topic))
            with tf.variable_scope('prob'):
                # this is the sampled gamma for this document, boosted to reduce the variance of the gradient
                # Me: shape augmentaion below not use: refer to section 5.1
                self.doc_vec = gamma_h_boosted(eps_alpha, u, self.alpha, self.B)
                self.doc_vec_beta = gamma_h_boosted(eps_beta, u, self.beta, self.B)

                # normalize
                self.doc_vec = tf.div(gam_alpha_beta, tf.reshape(tf.reduce_sum(gam_alpha_beta, 1), (-1, 1)))
                #self.doc_vec = tf.div(gam_alpha, tf.reshape(tf.reduce_sum(gam_alpha, 1), (-1, 1)))
                #self.doc_vec_beta = tf.div(gam_beta, tf.reshape(tf.reduce_sum(gam_beta, 1), (-1, 1)))
                # Me: doc_vec is now normalized z, the latent variable.
                self.doc_vec.set_shape(self.alpha.get_shape())
            # reconstruction
            # _NOTE THAT IF IT IS ORDINARY LDA, NO SOFTMAX in reconstruction
            # reconstruction
            # 1 means dimmension along which the log_softmax will be computed

            logits = tf.nn.log_softmax(tf.contrib.layers.batch_norm(
                utils.linear(self.doc_vec, self.vocab_size, scope='projection', no_bias=True)))
            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)

            # Compute KLD using samples
            dir1 = tf.contrib.distributions.Beta(self.prior_alpha, self.prior_beta)
            dir2 = tf.contrib.distributions.Beta(self.alpha, self.beta)
            #dir1 = tf.contrib.distributions.Dirichlet(self.prior_alpha)  # distribution one
            #dir2 = tf.contrib.distributions.Dirichlet(self.alpha)  # distribution two

            self.kld = dir2.log_prob(self.doc_vec) - dir1.log_prob(
                self.doc_vec)  # get the difference btw the distributions
            # Not used: max_kld_sampled = tf.arg_max(self.kld,0)
            # multiple samples part not consider
            self.kld = self.analytical_kld  #we ignore sampled kld because our analytical kld was stable

        self.analytical_objective = self.recons_loss * 0.8 + self.analytical_kld * 0.2
        self.objective = self.analytical_objective #self.recons_loss + self.warm_up * self.kld
        self.true_objective = self.analytical_objective #self.recons_loss + self.kld



        ########### Jointly training the encoder and decoder #################
        fullvars = tf.trainable_variables()

        ################### IF YOU WANT TO OPTIMIZE ENCODER AND DECODER DIFFERENTLY##################

        enc_vars = utils.variable_parser(fullvars, 'encoder')  # encoder trainable parameters
        dec_vars = utils.variable_parser(fullvars, 'decoder')  # Decoder trainable params

        # this is the standard gradient for the reconstruction network
        dec_grads = tf.gradients(self.objective, dec_vars)

        #####################################################
        # Now calculate the gradient for the encoding network#
        #####################################################

        # redefine kld and recons_loss for proper gradient back propagation
        # if self.n_sample ==1: No need bcls I use only one sample:
        # No reverse engineer again, we use the eps calculated already at the top
        # this is the sampled gamma for this document, boosted to reduce the variance of the gradient
        gammas = gamma_h_boosted(eps_alpha, u, self.alpha, self.B)
        # Me: doc_vec is unnormalized latent variable Z
        # normalize
        self.doc_vec = tf.div(gammas, tf.reshape(tf.reduce_sum(gammas, 1), (-1, 1)))
        # Me: Assert shape of doc_vec = alpha shape BUT NOT WORKING IN PYTORCH, worked in tf
        self.doc_vec.set_shape(self.alpha.get_shape())
        # Me: doc_vec is now normalized z, the latent variable.
        with tf.variable_scope("decoder", reuse=True):
            logits2 = tf.nn.log_softmax(tf.contrib.layers.batch_norm(
                utils.linear(self.doc_vec, self.vocab_size, scope='projection', no_bias=True)))
            self.recons_loss2 = -tf.reduce_sum(tf.multiply(logits2, self.x), 1)
            prior_sample = tf.squeeze(tf.random_gamma(shape=(1,), alpha=self.prior_alpha))
            prior_sample = tf.div(prior_sample, tf.reshape(tf.reduce_sum(prior_sample, 1), (-1, 1)))

            self.kld2 = tf.contrib.distributions.Beta(self.alpha, self.beta).log_prob(
                self.doc_vec) - tf.contrib.distributions.Beta(self.prior_alpha, self.prior_beta).log_prob(self.doc_vec)

        if analytical:
            kl_grad = tf.gradients(self.analytical_kld, enc_vars)
        else:
            kl_grad = tf.gradients(self.kld2, enc_vars)

        # this is the gradient we would use if the rejection sampler for the Gamma would always accept

        g_rep = tf.gradients(self.recons_loss2, enc_vars)

        # now define the gradient for the correction part

        # NB: In the reparametization here Note :alpha = self.alpha+tf.to_float(self.B)
        # 1. dh(epsilon, alpha, beta)##  dirivative of h(alpha, epsilon), derivative of EQ2

        # 2. gamma_h(epsilon, alpha,beta)## #Me: gamma_h is from equ_2: z=h_gamma(eps, alpha)= EQ2
        # Me: step_2 is z

        # 3.log_q means Log density of Gamma: log_q(z, alpha, beta) returns = gamma(alpha) + alpha*beta + (alpha - 1)logZ -Z*beta
        # Me: step_2 is z

        logpi_gradient = [tf.squeeze(separate_gradients(
            log_q(gamma_h(eps_alpha, self.alpha + tf.to_float(self.B), 1.), self.alpha + tf.to_float(self.B), 1.) + tf.log(
                dh(eps_alpha, self.alpha + tf.to_float(self.B), 1.)), var)) for var in enc_vars]

        # now multiply with the reconstruction loss
        reshaped1 = tf.reshape(self.recons_loss, (batch_size, 1))
        reshaped2 = tf.reshape(self.recons_loss, (batch_size, 1, 1))
        reshaped21 = tf.reshape(self.kld, (batch_size, 1))
        reshaped22 = tf.reshape(self.kld, (batch_size, 1, 1))

        g_cor = []
        #g_cor2 = []
        #g_cor2.append(tf.multiply(reshaped22, logpi_gradient[0]))
        #g_cor2.append(tf.multiply(reshaped21, logpi_gradient[1]))
        #g_cor2.append(tf.multiply(reshaped22, logpi_gradient[2]))
        #g_cor2.append(tf.multiply(reshaped21, logpi_gradient[3]))
        g_cor.append(tf.multiply(reshaped2, logpi_gradient[0]))
        g_cor.append(tf.multiply(reshaped1, logpi_gradient[1]))
        g_cor.append(tf.multiply(reshaped2, logpi_gradient[2]))
        g_cor.append(tf.multiply(reshaped1, logpi_gradient[3]))
        # sum over instances
        g_cor = [tf.reduce_sum(gc, 0) for gc in g_cor]
        #g_cor2 = [tf.reduce_sum(gc, 0) for gc in g_cor2]

        # finally sum up the three parts
        if not correction:
            enc_grads = [g_r + self.warm_up * g_e for g_r, g_c, g_e in zip(g_rep, g_cor, kl_grad)]
        else:
            #second correction term is not included
            enc_grads = [g_r + g_c + self.warm_up * g_e for g_r, g_c, g_e in zip(g_rep, g_cor, kl_grad)]
            #enc_grads = [g_r+g_c+g_c2+self.warm_up*g_e for g_r,g_c,g_c2,g_e in zip(g_rep,g_cor,g_cor2,kl_grad)]

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1,
                                           beta2=self.adam_beta2)
        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))
        self.optim_all = optimizer.apply_gradients(list(zip(enc_grads, enc_vars)) + list(zip(dec_grads, dec_vars)))


def log_dirichlet(x, alpha):
    first = -tf.reduce_sum(tf.lgamma(alpha), 1) + tf.lgamma(tf.reduce_sum(alpha, 1))
    second = tf.reduce_sum((alpha - 1.) * tf.log(x), 1)
    return first + second


"""
calculates the jacobian between a vector and some other tensor
"""


def jacobian(y_flat, x):
    n = tf.shape(y_flat)[0]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)
    return jacobian.stack()


"""
calculates the jacobian between a 2-dimensional matrix and some other tensor
"""


def jacobian2(y_flat, x):
    n = tf.shape(y_flat)[0]
    m = tf.shape(y_flat)[1]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]

    def body(j, result):
        loop_vars_inner_loop = [
            loop_vars[0],
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=m),
        ]
        _, _, row = tf.while_loop(lambda i, k, _: (k < m),
                                  lambda i, k, row: (i, k + 1, row.write(k, tf.gradients(y_flat[i][k], x))),
                                  loop_vars_inner_loop)
        result = result.write(j, row.stack())
        return (j + 1, result)

    _, jacobian = tf.while_loop(
        lambda j, _: (j < n),
        body,
        loop_vars)
    return jacobian.stack()


"""
returns the gradient for each data instance separately
"""


def separate_gradients(y_flat, x):
    n = tf.shape(y_flat)[0]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)
    return jacobian.stack()


# Log density of Ga(alpha, beta)
def log_q(z, alpha, beta):
    return -tf.lgamma(alpha) + alpha * tf.log(beta) \
           + (alpha - 1) * tf.log(z) - beta * z


# Log density of N(0, 1)
def log_s(epsilon):
    return -0.5 * tf.log(2 * tf.constant(math.pi)) - 0.5 * epsilon ** 2


# Transformation and its derivative
def gamma_h(epsilon, alpha, beta):
    """
    Reparameterization for gamma rejection sampler without shape augmentation.
    """
    b = alpha - 1. / 3.
    c = 1. / tf.sqrt(9. * b)
    v = 1. + epsilon * c

    return b * (v ** 3)


def gamma_h_boosted_B1(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = 1  # u.shape[1]
    K = alpha.shape[1]  # (batch_size,K)
    alpha_vec = alpha
    u_pow = tf.pow(u, 1. / alpha_vec) + 1e-10
    gammah = gamma_h(epsilon, alpha + B, 1.)
    return u_pow * gammah


def gamma_h_boosted(epsilon, u, alpha, model_B):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = tf.shape(u)[0]
    K = alpha.shape[1]  # (batch_size,K)
    r = tf.range(B)
    rm = tf.to_float(tf.reshape(r, [-1, 1, 1]))  # dim Bx1x1
    alpha_vec = tf.reshape(tf.tile(alpha, (B, 1)), (model_B, -1, K)) + rm  # dim BxBSxK + dim Bx1
    u_pow = tf.pow(u, 1. / alpha_vec) + 1e-10
    gammah = gamma_h(epsilon, alpha + tf.to_float(B), 1.)
    return tf.reduce_prod(u_pow, axis=0) * gammah


def gamma_grad_h(epsilon, alpha):
    """
    Gradient of reparameterization without shape augmentation.
    """
    b = alpha - 1. / 3.
    c = 1. / tf.sqrt(9. * b)
    v = 1. + epsilon * c

    return v ** 3 - 13.5 * epsilon * b * (v ** 2) * (c ** 3)


def dh(epsilon, alpha, beta):
    return (alpha - 1. / 3) * 3. / tf.sqrt(9 * alpha - 3.) * \
           (1 + epsilon / tf.sqrt(9 * alpha - 3)) ** 2 / beta


# Log density of proposal r(z) = s(epsilon) * |dh/depsilon|^{-1}
def log_r(epsilon, alpha, beta):
    return -tf.log(dh(epsilon, alpha, beta)) + log_s(epsilon)


# Density of the accepted value of epsilon
# (this is just a change of variables too)
def log_pi(eps, alpha):
    beta = 1.
    logq = log_q(gamma_h(eps, alpha, beta), alpha, beta)  # does not have to be boosted
    return log_s(eps) + \
           logq - \
           log_r(eps, alpha, beta)


def gamma_grad_logr(epsilon, alpha):
    """
    Gradient of log-proposal.
    """
    b = alpha - 1. / 3.
    c = 1. / tf.sqrt(9. * b)
    v = 1. + epsilon * c

    return -0.5 / b + 9. * epsilon * (c ** 3) / v


def gamma_grad_logq(epsilon, alpha):
    """
    Gradient of log-Gamma at proposed value.
    """
    h_val = gamma_h(epsilon, alpha)
    h_der = gamma_grad_h(epsilon, alpha)

    return tf.log(h_val) + (alpha - 1.) * h_der / h_val - h_der - tf.digamma(alpha)


def gamma_correction(epsilon, alpha):
    """
    Correction term grad (log q - log r)
    """
    return gamma_grad_logq(epsilon, alpha) - gamma_grad_logr(epsilon, alpha)


def calc_epsilon(gamma, alpha):
    return tf.sqrt(9. * alpha - 3.) * (tf.pow(gamma / (alpha - 1. / 3.), 1. / 3.) - 1.)


def train(sess, model,
          train_url,
          test_url,
          batch_size,
          vocab_size,
          analytical,
          alternate_epochs=1,  # 10
          lexicon=[],
          result_file='test.txt',
          B=1,
          warm_up_period=100):
    """train GDVAE model."""
    train_set, train_count = utils.data_set(train_url)
    test_set, test_count = utils.data_set(test_url)
    # hold-out development dataset
    train_size = len(train_set)
    validation_size = int(train_size * 0.1)
    dev_set = train_set[:validation_size]
    dev_count = train_count[:validation_size]
    train_set = train_set[validation_size:]
    train_count = train_count[validation_size:]
    optimize_jointly = True
    optimize_decoder = False
    dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)
    warm_up = 0
    min_alpha = 0.00001  #
    min_beta = 0.00002
    curr_B = B

    best_print_ana_ppx = 1e10
    early_stopping_iters = 5
    no_improvement_iters = 0
    stopped = False
    epoch = -1
    # for epoch in range(training_epochs):
    while not stopped:
        epoch += 1
        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
        if warm_up < 1.:
            warm_up += 1. / warm_up_period
        else:
            warm_up = 1.

        # -------------------------------
        # train
        if optimize_jointly:
            optim = model.optim_all
            print_mode = 'updating encoder and decoder'
        elif optimize_decoder:
            optim = model.optim_dec
            print_mode = 'updating decoder'
        else:
            optim = model.optim_enc
            print_mode = 'updating encoder'
        for i in range(alternate_epochs):
            loss_sum = 0.0
            ana_loss_sum = 0.0
            ppx_sum = 0.0
            kld_sum_train = 0.0
            ana_kld_sum_train = 0.0
            word_count = 0
            doc_count = 0
            recon_sum = 0.0
            for idx_batch in train_batches:
                data_batch, count_batch, mask = utils.fetch_data(
                    train_set, train_count, idx_batch, vocab_size)
                input_feed = {model.x.name: data_batch, model.mask.name: mask, model.keep_prob.name: 0.75,
                              model.warm_up.name: warm_up, model.min_alpha.name: min_alpha, model.min_beta.name: min_beta, model.B.name: curr_B}
                _, (loss, recon, kld_train, ana_loss, ana_kld_train) = sess.run((optim,
                                                                                 [model.true_objective,
                                                                                  model.recons_loss, model.kld,
                                                                                  model.analytical_objective,
                                                                                  model.analytical_kld]),
                                                                                input_feed)
                loss_sum += np.sum(loss)
                ana_loss_sum += np.sum(ana_loss)
                kld_sum_train += np.sum(kld_train) / np.sum(mask)  # mask is to get the size of the batch
                ana_kld_sum_train += np.sum(ana_kld_train) / np.sum(mask)
                word_count += np.sum(count_batch)
                # to avoid nan error
                count_batch = np.add(count_batch, 1e-12)
                # per document loss
                ppx_sum += np.sum(np.divide(loss, count_batch))
                doc_count += np.sum(mask)
                recon_sum += np.sum(recon)
            print_loss = recon_sum / len(train_batches)
            dec_vars = utils.variable_parser(tf.trainable_variables(), 'decoder')
            phi = dec_vars[0]
            phi = sess.run(phi)
            utils.print_top_words(phi, lexicon, result_file=None)
            print_ppx = np.exp(loss_sum / word_count)
            print_ana_ppx = np.exp(ana_loss_sum / word_count)
            print_ppx_perdoc = np.exp(ppx_sum / doc_count)
            print_kld_train = kld_sum_train / len(train_batches)
            print_ana_kld_train = ana_kld_sum_train / len(train_batches)
            print('| Epoch train: {:d} |'.format(epoch + 1),
                  print_mode, '{:d}'.format(i),
                  '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
                  '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
                  '| KLD: {:.5}'.format(print_kld_train),
                  '| Loss: {:.5}'.format(print_loss),
                  '| ppx anal.: {:.5f}'.format(print_ana_ppx),
                  '|KLD anal.: {:.5f}'.format(print_ana_kld_train))

        # -------------------------------
        # dev
        loss_sum = 0.0
        kld_sum_dev = 0.0
        ppx_sum = 0.0
        word_count = 0
        doc_count = 0
        recon_sum = 0.0
        print_ana_ppx = 0.0
        ana_loss_sum = 0.0
        for idx_batch in dev_batches:
            data_batch, count_batch, mask = utils.fetch_data(
                dev_set, dev_count, idx_batch, vocab_size)
            input_feed = {model.x.name: data_batch, model.mask.name: mask, model.keep_prob.name: 1.0,
                          model.warm_up.name: 1.0, model.min_alpha.name: min_alpha, model.min_beta.name: min_beta, model.B.name: B}
            loss, recon, kld_dev, ana_kld, ana_loss = sess.run(
                [model.objective, model.recons_loss, model.kld, model.analytical_kld, model.analytical_objective],
                input_feed)
            loss_sum += np.sum(loss)
            ana_loss_sum += np.sum(ana_loss)
            kld_sum_dev += np.sum(kld_dev) / np.sum(mask)
            word_count += np.sum(count_batch)
            count_batch = np.add(count_batch, 1e-12)
            ppx_sum += np.sum(np.divide(loss, count_batch))
            doc_count += np.sum(mask)
            recon_sum += np.sum(recon)
        print_ana_ppx = np.exp(ana_loss_sum / word_count)
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_kld_dev = kld_sum_dev / len(dev_batches)
        print_loss = recon_sum / len(dev_batches)
        if print_ppx < best_print_ana_ppx:
            no_improvement_iters = 0
            best_print_ana_ppx = print_ppx
            # check on validation set, if ppx better-> save improved model

            tf.train.Saver().save(sess, 'models/improved_model')

        else:
            no_improvement_iters += 1
            print('no_improvement_iters', no_improvement_iters, 'best ppx', best_print_ana_ppx)
            if no_improvement_iters >= early_stopping_iters:
                # if model has not improved for 30 iterations, stop training
                ###########STOP TRAINING############
                stopped = True
                print('stop training after', epoch, 'iterations,no_improvement_iters', no_improvement_iters)
                ###########LOAD BEST MODEL##########
                print('load stored model')
                tf.train.Saver().restore(sess, 'models/improved_model')
        print('| Epoch dev: {:d} |'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| KLD: {:.5}'.format(print_kld_dev),
              '| Loss: {:.5}'.format(print_loss))

        # -------------------------------
        # test
        if FLAGS.test:

            loss_sum = 0.0
            kld_sum_test = 0.0
            ppx_sum = 0.0
            word_count = 0
            doc_count = 0
            recon_sum = 0.0
            ana_loss_sum = 0.0
            ana_kld_sum_test = 0.0
            for idx_batch in test_batches:
                data_batch, count_batch, mask = utils.fetch_data(
                    test_set, test_count, idx_batch, vocab_size)
                input_feed = {model.x.name: data_batch, model.mask.name: mask, model.keep_prob.name: 1.0,
                              model.warm_up.name: 1.0, model.min_alpha.name: min_alpha, model.min_beta.name: min_beta, model.B.name: B}
                loss, recon, kld_test, ana_loss, ana_kld_test = sess.run(
                    [model.objective, model.recons_loss, model.kld, model.analytical_objective, model.analytical_kld],
                    input_feed)
                loss_sum += np.sum(loss)
                kld_sum_test += np.sum(kld_test) / np.sum(mask)
                ana_loss_sum += np.sum(ana_loss)
                ana_kld_sum_test += np.sum(ana_kld_test) / np.sum(mask)
                word_count += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)
                ppx_sum += np.sum(np.divide(loss, count_batch))
                doc_count += np.sum(mask)
                recon_sum += np.sum(recon)
            print_loss = recon_sum / len(test_batches)
            print_ppx = np.exp(loss_sum / word_count)
            print_ppx_perdoc = np.exp(ppx_sum / doc_count)
            print_kld_test = kld_sum_test / len(test_batches)
            print_ana_ppx = np.exp(ana_loss_sum / word_count)
            print_ana_kld_test = ana_kld_sum_test / len(train_batches)
            print('| Epoch test: {:d} |'.format(epoch + 1),
                  '| Perplexity: {:.9f}'.format(print_ppx),
                  '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
                  '| KLD: {:.5}'.format(print_kld_test),
                  '| Loss: {:.5}'.format(print_loss),
                  '| ppx anal.: {:.5f}'.format(print_ana_ppx),
                  '|KLD anal.: {:.5f}'.format(print_ana_kld_test))
            if stopped:
                # only do it once in the end
                print('calculate topic coherence (might take a few minutes)')
                coherence = utils.topic_coherence(test_set, phi, lexicon)
                print('topic coherence', str(coherence))


def myrelu(features):
    return tf.maximum(features, 0.0)

def parseArgs():
    # get line from config file
    args = sys.argv
    linum = int(args[1])
    argstring = ''
    configname = 'tfconfig'
    with open(configname, 'r') as rf:
        for i, line in enumerate(rf):
            # print(i,line)
            argstring = line
            if i + 1 == linum:
                print(line)
                break
    argparser = argparse.ArgumentParser()
    # define arguments
    argparser.add_argument('--adam_beta1', default=0.9, type=float)
    argparser.add_argument('--adam_beta2', default=0.999, type=float)
    argparser.add_argument('--learning_rate', default=1e-3, type=float)
    argparser.add_argument('--gd_prior_alpha', default=0.1, type=float)
    argparser.add_argument('--gd_prior_beta', default=0.1, type=float)
    argparser.add_argument('--B', default=1, type=int)
    argparser.add_argument('--n_topic', default=50, type=int)
    argparser.add_argument('--n_sample', default=1, type=int)
    argparser.add_argument('--warm_up_period', default=100, type=int)
    argparser.add_argument('--nocorrection', action="store_true")
    argparser.add_argument('--data_dir', default='data/20news', type=str)
    return argparser.parse_args(argstring.split())


def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
        non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
        non_linearity = tf.nn.sigmoid
    else:
        non_linearity = myrelu

    analytical = True
    args = parseArgs()
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    learning_rate = args.learning_rate
    gd_prior_alpha = args.gd_prior_alpha
    gd_prior_beta = args.gd_prior_beta
    B = args.B
    warm_up_period = args.warm_up_period
    n_sample = args.n_sample
    n_topic = args.n_topic
    lexicon = []
    vocab_path = os.path.join(args.data_dir, 'vocab.new')
    with open(vocab_path, 'r') as rf:
        for line in rf:
            word = line.split()[0]
            lexicon.append(word)
    vocab_size = len(lexicon)

    dvae = DVAE(analytical=analytical,
                vocab_size=vocab_size,
                n_hidden=FLAGS.n_hidden,
                n_topic=n_topic,
                n_sample=n_sample,
                learning_rate=learning_rate,
                batch_size=FLAGS.batch_size,
                non_linearity=non_linearity,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                B=B,
                gd_prior_alpha=gd_prior_alpha,
                gd_prior_beta=gd_prior_beta,
                correction=(not args.nocorrection))
    sess = tf.Session()
    init = tf.global_variables_initializer()
    result = sess.run(init)
    train_url = os.path.join(args.data_dir, 'train.feat')
    test_url = os.path.join(args.data_dir, 'test.feat')

    train(sess, dvae, train_url, test_url, FLAGS.batch_size, vocab_size, analytical, lexicon=lexicon,
          result_file=None, B=B,
          warm_up_period=warm_up_period)


if __name__ == '__main__':
    app.run(main)
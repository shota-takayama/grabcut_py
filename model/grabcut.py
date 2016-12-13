import numpy as np
import sklearn.mixture.gaussian_mixture as gm
from graph_tool.all import *
import cv2

class GrabCut(object):

    def __init__(self, K, gamma):
        self.K = K
        self.gamma = gamma


    def segment(self, src, fore, T, dname):
        self.__init_params(src, fore)
        height, width, _ = src.shape
        for t in range(T):
            print 'iter: {}'.format(t)
            gmm = [gm.GaussianMixture(self.K) for i in range(2)]
            pi, gaussf = self.__gmm_params(gmm, src)

            g = graph_tool.Graph()
            eprop = g.new_edge_property('float')
            self.__build_graph(g, eprop, gaussf, pi, src)
            res = graph_tool.flow.edmonds_karp_max_flow(g, g.vertex(0), g.vertex(1), eprop)
            self.__update_alpha(g, res, width)
            self.__save_segmented_image(src, '{0}/seg{1}.png'.format(dname, t))

        return self.__segment_image(src)


    def __init_params(self, src, fore):
        height, width = src.shape[:2]
        tl_x, tl_y, br_x, br_y = fore
        self.alpha = np.zeros((height, width)).astype('uint')
        self.alpha[tl_y:br_y, tl_x:br_x] = 1
        kai = (np.sum((src[:-1, :] - src[1:, :]) ** 2) + np.sum((src[:, :-1] - src[:, 1:]) ** 2)) / (height * width)
        self.beta = 0.5 / kai


    def __gaussian(self, mu, cov):
        def f(x):
            numer = np.exp(-0.5 * (x - mu).T.dot(np.linalg.inv(cov).dot(x - mu)))
            denom = (2 * np.pi) ** 1.5 * np.sqrt(np.linalg.det(cov))
            return numer / denom
        return f


    def __gmm_params(self, gmm, src):
        pi, gaussf = [[], []], [[], []]
        for i in range(2):
            gmm[i].fit(src[self.alpha == i])
            pi[i] = gmm[i].weights_
            gaussf[i] = [self.__gaussian(gmm[i].means_[k], gmm[i].covariances_[k]) for k in range(self.K)]
        return pi, gaussf


    def __d(self, src):
        return src[:-1, :] - src[1:, :], src[:, :-1] - src[:, 1:]


    def __a(self, alpha):
        return alpha[:-1, :] - alpha[1:, :], alpha[:, :-1] - alpha[:, 1:]


    def __likelihood(self, gaussf, pi, pixel):
        return -np.log(gaussf(pixel)) - np.log(pi)


    def __difference(self, pixel_diff):
        return self.gamma * np.exp(-self.beta * np.linalg.norm(pixel_diff) ** 2)


    def __set_value(self, g, eprop):
        def f(i1, i2, val):
            eprop[g.add_edge(g.vertex(i1), g.vertex(i2))] = val
        return f


    def __set_likelihood(self, set_func, gaussf, pi, pixel, ind):
        likel = [np.min([self.__likelihood(gaussf[i][k], pi[i][k], pixel) for k in range(self.K)]) for i in range(2)]
        set_func(0, ind, likel[0])
        set_func(ind, 1, likel[1])


    def __set_difference(self, set_func, d, a, ind, stride):
        diff = self.__difference(d) if a else 0.0
        set_func(ind, ind + stride, diff)
        set_func(ind + stride, ind, diff)


    def __build_graph(self, g, eprop, gaussf, pi, src):
        height, width = src.shape[:2]
        g.add_vertex(height * width + 2) # 0: source, 1: sink
        dy, dx = self.__d(src)
        ay, ax = self.__a(self.alpha)
        for y in range(height):
            for x in range(width):
                ind = y * width + x + 2
                set_func = self.__set_value(g, eprop)
                self.__set_likelihood(set_func, gaussf, pi, src[y, x], ind)
                if y < height - 1:
                    self.__set_difference(set_func, dy[y, x], ay[y, x], ind, width)
                if x < width - 1:
                    self.__set_difference(set_func, dx[y, x], ax[y, x], ind, 1)


    def __update_alpha(self, g, res, w):
        inds = []
        for e in g.vertex(1).in_edges():
            if res[e] != 0.0:
                inds += [int(str(e.source())) - 2]
        inds = np.array(inds)
        self.alpha[inds / w, inds % w] = 0


    def __segment_image(self, src):
        dst = np.copy(src).astype('uint8')
        dst[self.alpha == 0] = 255
        return dst


    def __save_segmented_image(self, src, fname):
        dst = self.__segment_image(src)
        cv2.imwrite(fname, dst)

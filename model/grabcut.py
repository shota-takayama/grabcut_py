import numpy as np
import sklearn.mixture.gaussian_mixture as gm
from graph_tool.all import *
import cv2

class GrabCut(object):

    def __init__(self, K, gamma):
        self.K = K
        self.gamma = gamma


    def segment(self, src, tl, br, T, dname):
        self.__init_params(src, tl, br)
        height, width, _ = src.shape
        self.__save_segmented_image(src, '{0}/seg{1}.png'.format(dname, 0))
        for t in range(T):
            print 'iter: {}'.format(t + 1)
            g = graph_tool.Graph()
            eprop = g.new_edge_property('float')
            gmm = [gm.GaussianMixture(self.K) for i in range(2)]
            pi, gaussf = self.__gmm_params(gmm, src)

            res = self.__solve_graph(g, eprop, gaussf, pi, src)
            self.__update_alpha(g, res, width)
            self.__save_segmented_image(src, '{0}/seg{1}.png'.format(dname, t + 1))

        return self.__segment_image(src)


    def __init_alpha(self, height, width, tl, br):
        self.alpha = np.zeros((height, width)).astype('uint')
        self.alpha[tl[0]:br[0], tl[1]:br[1]] = 1


    def __init_beta(self, src, height, width):
        dy, dx = self.__shift(src)
        kai = (np.sum(dy ** 2) + np.sum(dx ** 2)) / (2 * height * width - height - width)
        self.beta = 0.5 / kai


    def __init_params(self, src, tl, br):
        height, width = src.shape[:2]
        self.__init_alpha(height, width, tl, br)
        self.__init_beta(src, height, width)


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


    def __shift(self, narray):
        return narray[:-1, :] - narray[1:, :], narray[:, :-1] - narray[:, 1:]


    def __likelihood(self, gaussf, pi, color):
        return -np.log(gaussf(color)) - np.log(pi)


    def __difference(self, color_diff):
        return self.gamma * np.exp(-self.beta * np.linalg.norm(color_diff) ** 2)


    def __set_value(self, g, eprop):
        def f(i1, i2, val):
            eprop[g.add_edge(g.vertex(i1), g.vertex(i2))] = val
        return f


    def __set_likelihood(self, setter, gaussf, pi, color, ind):
        likel = [np.min([self.__likelihood(gaussf[i][k], pi[i][k], color) for k in range(self.K)]) for i in range(2)]
        setter(0, ind, likel[0])
        setter(ind, 1, likel[1])


    def __set_difference(self, setter, d, a, ind, stride):
        diff = self.__difference(d) if a else 0.0
        setter(ind, ind + stride, diff)
        setter(ind + stride, ind, diff)


    def __build_graph(self, g, eprop, gaussf, pi, src):
        height, width = src.shape[:2]
        g.add_vertex(height * width + 2) # 0: source, 1: sink
        dy, dx = self.__shift(src)
        ay, ax = self.__shift(self.alpha)
        for y in range(height):
            for x in range(width):
                ind = y * width + x + 2
                setter = self.__set_value(g, eprop)
                self.__set_likelihood(setter, gaussf, pi, src[y, x], ind)
                if y < height - 1:
                    self.__set_difference(setter, dy[y, x], ay[y, x], ind, width)
                if x < width - 1:
                    self.__set_difference(setter, dx[y, x], ax[y, x], ind, 1)


    def __solve_graph(self, g, eprop, gaussf, pi, src):
        self.__build_graph(g, eprop, gaussf, pi, src)
        return graph_tool.flow.edmonds_karp_max_flow(g, g.vertex(0), g.vertex(1), eprop)


    def __update_alpha(self, g, res, width):
        inds = []
        for e in g.vertex(1).in_edges():
            if res[e] != 0.0:
                inds += [int(str(e.source())) - 2]
        inds = np.array(inds)
        self.alpha[inds / width, inds % width] = 0


    def __segment_image(self, src):
        dst = np.copy(src).astype('uint8')
        dst[self.alpha == 0] = 255
        return dst


    def __save_segmented_image(self, src, fname):
        dst = self.__segment_image(src)
        cv2.imwrite(fname, dst)

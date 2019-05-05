from PIL import Image
import numpy as np
import logging
logger = logging.getLogger('BoneAge')


crop_bone_name = ['wan', 'zhang1', 'zhang35', 'chi', 'rao', 'zhi', 'zi']

crop_bone_counter = {
    'zi': 1,
    'wan': 1,
    'chi': 1,
    'rao': 1,
    'zhang1': 1,
    'zhang35': 2,
    'zhi': 8,
}


class CropBone(object):
    """docstring for CropBone"""

    def __init__(self, img, points):
        super(CropBone, self).__init__()
        self.img = img
        self.points = points

    def rotateSpecificDegree(self, img, points, degree):
        deg = degree

        ct = np.array(img.size, dtype=np.float) / 2
        npts = points - ct
        rotrad = np.deg2rad(deg)
        rot = np.array([[np.cos(rotrad), -np.sin(rotrad)],
                        [np.sin(rotrad), np.cos(rotrad)]])
        npts = np.dot(npts, rot)

        nimg = img.rotate(deg, Image.BICUBIC, expand=True)
        npts += np.array(nimg.size, dtype=np.float) / 2

        return nimg, npts

    def len_name_gu(self, name):
        return crop_bone_counter[name]

    def get_name_gu(self, name):
        assert name in crop_bone_name
        if name == 'zi':
            return self.get_zi_gu()
        if name == 'wan':
            return self.get_wan_gu()
        if name == 'chi':
            return self.get_chi_gu()
        if name == 'rao':
            return self.get_rao_gu()
        if name == 'zhang35':
            return self.get_zhang35_gu()
        if name == 'zhang1':
            return self.get_zhang1_gu()
        if name == 'zhi':
            return self.get_zhi_gu()
        return []

    def get_zi_gu(self):
        ar1 = np.array(self.points[41:45])
        ar2 = np.array(self.points[46:49])
        ct1 = (ar1[0] + ar1[1] + ar1[2] + ar1[3]) / 4
        ct2 = (ar2[0] + ar2[1] + ar2[2]) / 3
        ct = (ct1 + ct2) / 2
        x, y = (ct1 - ct)
        pts = [ct1, ct, [ct[0] + y, ct[1] - x], [ct1[0] + y, ct1[1] - x]]
        rad = np.arctan2(y, x)
        deg = np.rad2deg(rad)
        rtn_img, rtn_pts = self.rotateSpecificDegree(self.img, pts, deg)

        rtn = [(rtn_img, rtn_pts, 1)]
        return rtn

    def get_wan_gu(self):
        pts = np.concatenate([self.points[49:51],
                              self.points[57:59],
                              self.points[15:18],
                              self.points[46:48]]
                             )
        A = np.average(pts[0:2], axis=0)
        E = np.average(pts[2:4], axis=0)
        X = np.average(pts[7:], axis=0)
        b = (E - A) / np.linalg.norm(E - A)
        B = np.dot(X - A, b) * b + A
        V2 = (X - B)
        h2 = np.sqrt(V2[0] * V2[0] + V2[1] * V2[1]) * 1.2
        U = (B - A)
        w = np.sqrt(U[0] * U[0] + U[1] * U[1]) * 1.11
        h = w * 0.8
        if h2 > h:
            h = h2
            w = h * 1.25
        A = (A - B) / np.linalg.norm(A - B) * w + B
        C = (X - B) / np.linalg.norm(X - B) * h + B
        D = A + C - B

        dx, dy = B - A
        deg = np.rad2deg(np.arctan2(dy, dx))
        rtn_img, rtn_pts = self.rotateSpecificDegree(
            self.img, [A, B, C, D], deg)
        rtn = [(rtn_img, rtn_pts, 1. * h / w)]
        return rtn

    '''
        direction_pts 定方向的点集
        tip_id 方向向量朝向点集中的哪个点, 示例见draw_debug.py
        w_expend, h_expend 宽高扩大的倍数
        h_w_ratio 高宽比
    '''

    def get_pca_crop_pts(self, direction_pts, tip_id, crop_pts, h_w_ratio, w_expend=1.2, h_expend=1.2):

        def getDirection1(rd, tipId):
            rd = np.array(rd) * [1.0, -1.0]
            crd = rd

            avg = crd.mean(axis=0)
            ctc = crd - avg

            _, __, v = np.linalg.svd(np.dot(ctc.T, ctc))

            direction = v[0]
            if np.dot(v[0], rd[tipId] - avg) < 0.0:
                direction = -direction

            return direction

        pts = np.array(crop_pts)
        V = np.array(getDirection1(direction_pts, tip_id))
        rad = np.deg2rad(-np.rad2deg(np.arctan2(V[1], V[0])))
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        npts = [np.array([0, 0])]
        for i in range(1, len(pts)):
            npts.append(pts[i] - pts[0])
        npts = np.array(npts)
        npts = np.dot(npts, rot)
        lbV = npts.min(axis=0)
        rtV = npts.max(axis=0)
        w = rtV[0] - lbV[0]
        h = rtV[1] - lbV[1]
        nh = h * h_expend
        nw = w * w_expend
        if h_w_ratio:
            if nh / nw > h_w_ratio:
                nw = nh / h_w_ratio
            else:
                nh = nw * h_w_ratio
        lbV[0] -= (nw - w) / 2.0
        rtV[0] += (nw - w) / 2.0
        lbV[1] -= (nh - h) / 2.0
        rtV[1] += (nh - h) / 2.0
        ltV = np.array([lbV[0], rtV[1]])
        rbV = np.array([rtV[0], lbV[1]])
        npts = np.array([ltV, rtV, rbV, lbV])
        rad_r = -rad
        rot = np.array([[np.cos(rad_r), -np.sin(rad_r)], [np.sin(rad_r), np.cos(rad_r)]])
        npts = np.dot(npts, rot)
        for i in range(len(npts)):
            npts[i] = npts[i] + pts[0]
        rtn_img, rtn_pts = self.rotateSpecificDegree(
            self.img, npts, np.rad2deg(rad))
        rtn = [(rtn_img, rtn_pts, 1. * nh / nw)]
        return rtn

    def get_chi_gu(self):
        return self.get_pca_crop_pts(self.points[49:59], -1, self.points[49:54], h_w_ratio=1.2, w_expend=1.4, h_expend=2.8)

    def get_rao_gu(self):
        return self.get_pca_crop_pts(self.points[49:59], -1, self.points[54:59], h_w_ratio=0.8, w_expend=1.36, h_expend=2.)

    def get_zhang35_gu(self):
        zhang3 = self.get_pca_crop_pts(self.points[28:36], -1, self.points[33:36], h_w_ratio=1., w_expend=1.8, h_expend=1.8)
        zhang5 = self.get_pca_crop_pts(self.points[10:18], -1, self.points[15:18], h_w_ratio=1., w_expend=1.8, h_expend=1.8)
        return zhang3 + zhang5

    def get_zhang1_gu(self):
        zhang1 = self.get_pca_crop_pts(self.points[41:49], -1, self.points[46:49], h_w_ratio=0.8, w_expend=3., h_expend=1.5)
        return zhang1

    def get_zhi_gu(self, h_w_ratio=1., w_expend=3., h_expend=1.5):
        rtn = []
        # 7
        rtn.extend(self.get_pca_crop_pts(self.points[36:41] + self.points[45:46], -1,
                                         self.points[37:41] + self.points[45:46], h_w_ratio, w_expend, h_expend))
        # 8
        rtn.extend(self.get_pca_crop_pts(self.points[40:46], -2,
                                         self.points[41:45], h_w_ratio, w_expend, h_expend))
        # 12
        rtn.extend(self.get_pca_crop_pts(self.points[18:23] + self.points[26:27], -1,
                                         self.points[19:23] + self.points[26:27], h_w_ratio, w_expend, h_expend))
        # 13
        rtn.extend(self.get_pca_crop_pts(self.points[22:28] + self.points[32:33], -1,
                                         self.points[23:26] + self.points[27:28] + self.points[32:33], h_w_ratio, w_expend, h_expend))
        # 14
        rtn.extend(self.get_pca_crop_pts(self.points[27:33], -2,
                                         self.points[28:32], h_w_ratio, w_expend, h_expend))
        # 18
        rtn.extend(self.get_pca_crop_pts(self.points[0:5] + self.points[8:9], -1,
                                         self.points[1:5] + self.points[8:9], h_w_ratio, w_expend, h_expend))
        # 19
        rtn.extend(self.get_pca_crop_pts(self.points[4:10] + self.points[14:15], -1,
                                         self.points[5:8] + self.points[9:10] + self.points[14:15], h_w_ratio, w_expend, h_expend))
        # 20
        rtn.extend(self.get_pca_crop_pts(self.points[9:15], -2,
                                         self.points[10:14], h_w_ratio, w_expend, h_expend))
        return rtn

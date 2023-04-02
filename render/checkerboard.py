"""
checkerboard class suitable for nueral_renderer

Author: Xianghui Xie
Date: March 29, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
from psbody.mesh import Mesh
import torch
import numpy as np


class CheckerBoard:
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white)/255.
        self.black = np.array(black)/255.
        self.verts, self.faces, self.texts = None, None, None
        self.offset = None
        self.checker_mesh = None

    def init_checker(self, offset, plane='xz', xlength=50, ylength=50, square_size=0.5):
        "generate checkerboard and prepare v, f, t"
        # checker = self.gen_checker_xy(self.black, self.white, square_size, xlength, ylength)
        checker = self.gen_checker_xy_no_repeat(self.black, self.white, square_size, xlength, ylength)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplemented
        elif plane == 'xy':
            # no transformation
            pass

        checker.v = np.matmul(checker.v, rot.T)
        self.checker_mesh = checker

        # apply offsets
        checker.v += offset
        self.offset = offset

        self.verts, self.faces, self.texts = self.prep_checker_rend(checker)

    def get_rends(self):
        return self.verts, self.faces, self.texts

    def append_checker(self, checker):
        "append another checker"
        v, f, t = checker.get_rends()
        nv = self.verts.shape[1]
        self.verts = torch.cat([self.verts, v], 1)
        self.faces = torch.cat([self.faces, f+nv], 1)
        self.texts = torch.cat([self.texts, t], 1)

    @staticmethod
    def gen_checkerboard(white, black, square_size=0.5, total_size=5.0, plane='xz'):
        "plane: the checkboard is in parallal to which plane"
        checker = CheckerBoard.gen_checker_xy_no_repeat(white, black, square_size, total_size, total_size)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90, so that the checker plane is perpendicular to y-axis
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
            # now the plane is from (0, 0, 0) to (size, 0, -size)
        elif plane == 'yz':
            raise NotImplemented
        checker.v = np.matmul(checker.v, rot)
        return checker

    def prep_checker_rend(self, checker:Mesh):
        verts = torch.from_numpy(checker.v.astype(np.float32)).cuda().unsqueeze(0)
        faces = torch.from_numpy(checker.f.astype(int)).cuda().unsqueeze(0)
        nf = checker.f.shape[0]
        # texts = torch.zeros(1, nf, 4, 4, 4, 3).cuda()
        texts = torch.zeros(1, nf, 1, 1, 1, 3).cuda()
        for i in range(nf):
            texts[0, i, :, :, :, :] = torch.tensor(checker.fc[i], dtype=torch.float32).cuda()
        return verts, faces, texts

    @staticmethod
    def gen_checker_xy_no_repeat(black, white, square_size=0.5, xlength=5.0, ylength=5.0, vc=False):
        """
        generate a chckerborad parallel to x-y plane, with a normal direction in +z axis
        no repeated vertices
        diagram for vertex index: i --index in x direction, j--index in y direction
            ^
            | y-axis
            |
            (i, j+1) --- (i+1, j+1)
            |         /     |
            |       /       |
            |     /         |
            (i, j) ----- (i+1, j) ---> x-axis
        Args:
            black: RGB color for the black squares
            white: RGB color for the white squares
            square_size: size of a square
            xlength: total length of the checkerboard
            ylength:
            vc:

        Returns: a psbody Mesh instance, including vertc, faces, and face (square) colors


        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts_count = (xsquares+1) * (ysquares+1) # total number of vertices in this board
        x, y= np.arange(0, (xsquares+1)*square_size, square_size), np.arange(0, (ysquares+1)*square_size, square_size)
        if len(x) > xsquares + 1:
            x = x[:xsquares+1]
        if len(y) > ysquares + 1:
            y = y[:ysquares+1]
        xv, yv = np.meshgrid(x, y)
        verts = np.stack((xv, yv, np.zeros_like(xv)), -1) # (XL, YL, 3)
        verts_all = verts.reshape((verts_count, 3)) # (L, 3), first XL elements have zero y values,

        faces, texts = [], []
        for i in range(xsquares):
            for j in range(ysquares):
                # decide the color and vertex index for each triangle face
                # get the index in full array
                p0i = j*(xsquares + 1) + i
                p1i = (j+1)*(xsquares + 1) + i + 1
                p2i = (j+1)*(xsquares + 1) + i
                faces.append(np.array([p0i, p1i, p2i]))

                # the second triangle
                # get the index in full array
                p0i = j * (xsquares + 1) + i
                p1i = j * (xsquares + 1) + i + 1
                p2i = (j + 1) * (xsquares + 1) + i + 1
                faces.append(np.array([p0i, p1i, p2i]))

                if (i + j) % 2 == 0:
                    texts.append(black)
                    texts.append(black)
                else:
                    texts.append(white)
                    texts.append(white)
        mesh = Mesh(v=verts_all, f=np.array(faces), fc=np.array(texts))
        return mesh

    @staticmethod
    def from_meshes(meshes, yaxis_up=True, xlength=50, ylength=20):
        """
        initialize checkerboard ground from meshes
        """
        vertices = [x.v for x in meshes]
        if yaxis_up:
            # take ymin
            y_off = np.min(np.concatenate(vertices, 0), 0)
        else:
            # take ymax
            y_off = np.min(np.concatenate(vertices, 0), 0)
        offset = np.array([xlength/2, y_off[1], ylength/2]) # center to origin
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength)
        return checker

    @staticmethod
    def from_verts(verts, yaxis_up=True, xlength=5, ylength=5, square_size=0.2):
        """
        verts: (1, N, 3)
        set up a ground plane(parallel to x-z plane) based on the verts
        if y-axis is up, the plane will have y value of the minimum y, otherwise it takes maximum y value
        """
        if yaxis_up:
            y_off = torch.min(verts[0], 0)[0].cpu().numpy()
        else:
            y_off = torch.max(verts[0], 0)[0].cpu().numpy()
        offset = np.array([-xlength/2, y_off[1], -ylength/2])
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength, square_size=square_size)
        return checker


def test():
    """
    generate a checkerboard and use psbody.mv to visualize
    usage: python checkboard.py
    """
    from psbody.mesh import MeshViewer
    generator = CheckerBoard()
    checker = generator.gen_checker_xy_no_repeat(generator.black, generator.white, vc=True)
    checker.write_ply('debug/checkerboard.ply')
    mv = MeshViewer()
    mv.set_static_meshes([checker])


def test2():
    from psbody.mesh import MeshViewer
    generator = CheckerBoard()
    checker = generator.gen_checker_xy_no_repeat(generator.black, generator.white, vc=False)
    checker.write_ply('debug/meshes/checker_norepeat.ply')
    mv = MeshViewer()
    mv.set_static_meshes([checker])


if __name__ == '__main__':
    # test()
    test2()
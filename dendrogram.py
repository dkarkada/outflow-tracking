import numpy as np

def make_dendrogram(img):
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    im = np.ones((img.shape[0]+2, img.shape[1]+2))
    im[1:-1, 1:-1] = img
    im = np.round(im * 255)

    visited = np.ones(im.shape)
    visited[1:-1, 1:-1] = 0
    
    ind = np.unravel_index(np.argmax(im), im.shape)
    root, _ = dendrogram(im[ind], ind, im, visited)
    return Dendrogram(root, im[1:-1, 1:-1])
    
def find_min(start, im):
    visited = np.ones(im.shape)
    visited[1:-1, 1:-1] = 0
    visited[start] = 1
    cur = im[start]
    min_ind = start
    bfs_queue = {*neighbors(*start)}
    while bfs_queue:
        r, c = bfs_queue.pop()
        if visited[r, c]:
            continue
        visited[r, c] = 1            
        if im[r, c] < cur:
            cur = im[r, c]
            min_ind = (r, c)
            bfs_queue = {*neighbors(r, c)}
        if im[r, c] == cur:
            for n in neighbors(r, c):
                if not visited[n]:
                    bfs_queue.add(n)
    return min_ind, cur
    
def neighbors(r, c):
    return [(r-1, c), (r+1, c), (r, c-1), (r, c+1),
            (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
    
def dendrogram(toplevel, start, im, visited):
    (r, c), level = find_min(start, im)
    im_crop = im[1:-1, 1:-1]
    d = Dendro(level, im_crop)
    
    accrete_history = []
    bfs_queue = {(r, c)}
    while level < toplevel:
        failed = set()
        accreted = set()
        siblings = []
        while bfs_queue:
            r, c = bfs_queue.pop()
            if visited[r, c]:
                continue
            if im[r, c] == level:
                visited[r, c] = 1
                accreted.add((r-1, c-1))
                for n in neighbors(r, c):
                    if not visited[n]:
                        bfs_queue.add(n)
            elif im[r, c] < level:
                sibling, surround = dendrogram(level, (r, c), im, visited)
                bfs_queue = bfs_queue.difference(sibling.region)
                bfs_queue = bfs_queue.union(surround)
                siblings.append(sibling)
            else:
                failed.add((r, c))
        
        
        if siblings:
            next_d = Dendro(level, im_crop)
            d.initialize(accrete_history, toplevel=level)
            accrete_history = []
            d.parent = next_d
            next_d.children.append(d)
            for s in siblings:
                s.parent = next_d
                next_d.children.append(s)
            d = next_d
        
        accrete_history.append(accreted)            
        level += 1
        bfs_queue = failed
    
    d.initialize(accrete_history, toplevel)
    return d, bfs_queue

class Dendrogram:
    
    def __init__(self, root, im):
        self.root = root
        # HYPERPARAM
        root.merge(7)
        root.calculate()
        assert im.shape[0] == im.shape[1]
        self.im = im
        self.boundary = im.shape[0] // 2
        self.branches = root.descendants
        self.N = len(self.branches)
        for n in range(self.N):
            self.branches[n].id = n
        
        self.metric = np.zeros((self.N, self.N))
        self.compute_metric(root)
        
        self.hierarchy = np.zeros((self.N, self.N))
        for branch_id in range(self.N):
            twigs = self.branches[branch_id].descendants
            for twig_id in [t.id for t in twigs]:
                self.hierarchy[branch_id, twig_id] = 1
        
        mass_arr = [b.mass for b in self.branches]
        # append source mass
        self.masses = np.array(mass_arr + [max(mass_arr)])
        self.x = np.array([b.x for b in self.branches])
        
    def compute_metric(self, branch):
        if len(branch.children) == 0:
            return
        
        subtrees = []
        for c in branch.children:
            self.compute_metric(c)
            subtree = c.descendants
            for b in subtree:
                dist = 1 + self.metric[c.id, b.id]
                assert self.metric[branch.id, b.id] == 0
                self.metric[branch.id, b.id] = dist
                self.metric[b.id, branch.id] = dist
            subtrees.append((c, subtree))
            
        for c1_ind in range(len(subtrees)):
            for c2_ind in range(c1_ind+1, len(subtrees)):
                c1, subtree1 = subtrees[c1_ind]
                c2, subtree2 = subtrees[c2_ind]
                for b1 in subtree1:
                    for b2 in subtree2:
                        dist = 2 + self.metric[c1.id, b1.id] + self.metric[c2.id, b2.id]
                        assert self.metric[b1.id, b2.id] == 0
                        self.metric[b1.id, b2.id] = dist
                        self.metric[b2.id, b1.id] = dist        

class Dendro:
    
    def __init__(self, level, im):
        self.im = im
        self.map = np.zeros(im.shape)
        self.orig_level = level
        self.children = []
        self.descendants = [self]
        self.parent = None
        
        self.region = None
        self.mass = None
        self.total_mass = None
        self.mass_frac = None
        self.x = None
        
        self.covariance = None
        
        self.id = None
        self.traj_id = None
        
    def initialize(self, accrete_hist, toplevel):
        # important: children are initialized first            
        depth = toplevel - self.orig_level
        assert depth == len(accrete_hist)
        self.region = set().union(*accrete_hist)\
                           .union(*[c.region for c in self.children])
        
        for px in self.region:
            px_depth = min(depth, toplevel - self.im[px])
            self.map[px] += px_depth
    
    def merge(self, depth_thresh):
        def do_merge(c):
            self.map += c.map
            self.region = self.region.union(c.region)
            for grandchild in c.children:
                grandchild.parent = self
                self.children.append(grandchild)
                
        def should_merge(c):
            if len(c.children) == 0:
                return np.max(c.map) < depth_thresh

        to_remove = []
        new_children = []
        for c in self.children.copy():
            c.merge(depth_thresh)                
            if np.max(c.map) < depth_thresh:
                do_merge(c)
                to_remove.append(c)
        self.children = [c for c in self.children if c not in to_remove]
        
        if len(self.children) == 1:
            c = self.children[0]
            self.children = []
            do_merge(c)            
        
        for c in self.children:
            self.descendants += c.descendants
            
    def calculate(self):
        def px_to_coord(px):
            x = px[1] - self.im.shape[1]//2
            y = self.im.shape[0]//2 - px[0]
            return np.array([x, y])
        
        for c in self.children:
            c.calculate()
            
        x_vec = np.zeros(2)
        for px in np.ndindex(*self.map.shape):
            x_vec += self.map[px] * px_to_coord(px)
        self.mass = self.map.sum()
        # COMPLETE HACK
        self.mass = max(1, self.mass)
        self.total_mass = self.mass + sum([c.mass for c in self.children])
        # center of mass of exclusive mass
        self.x = x_vec / self.mass
        self.mass_frac = self.mass / self.total_mass
        
        # covariance matrix calculation
        cov_mat = np.zeros((2, 2))
        for px in np.ndindex(*self.map.shape):
            px_vec = px_to_coord(px) - self.x
            cov_mat += self.map[px] * np.outer(px_vec, px_vec)
        self.covariance = cov_mat / self.mass

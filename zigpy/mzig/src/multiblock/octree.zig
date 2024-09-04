const std = @import("std");

pub const BlockError = error{
    BlockHasChildrenAlready,
    BlockHasParentAlready,
    ChildrenNotFoundInParent,
    QuadrNotExists,
    InvalidNeighbour,
    BlockAllocationError,
};

fn all_possible_neighbours(comptime dim: i32) [if (dim == 2) 8 else 26][dim]i8 {
    var offsets: [if (dim == 2) 8 else 26][dim]i8 = undefined;
    var idx: usize = 0;
    const valid_offsets: [3]i8 = .{ -1, 0, 1 };
    const null_dim: [1]i8 = .{0};
    for (valid_offsets) |x| {
        for (valid_offsets) |y| {
            for (if (dim == 3) valid_offsets else null_dim) |z| {
                const p: [dim]i8 = if (dim == 2) .{ x, y } else .{ x, y, z };
                if (x == 0 and y == 0 and z == 0) {
                    continue;
                }
                offsets[idx] = p;
                idx += 1;
            }
        }
    }
    std.debug.assert((if (dim == 2) 8 else 26) == idx); // internal error
    return offsets;
}

pub fn OctreeBlock(comptime dim: i32, T: type) type {
    if (dim != 2 and dim != 3) {
        std.debug.panic("Only blocks of 2 or 3 dimensions are possible", .{});
    }

    return extern struct {
        const Self = @This();
        pub const n_children = 1 << dim;

        /// Parent block, None for level 0
        parent: ?*Self,
        /// Block may have no children or 2^dim children blocks
        children: [n_children]?*Self,
        /// Neighbour blocks to this, as [min, max][x, y, z]
        adj_neighbours: [dim][2]?*Self,
        data: T,

        /// index for quadrants positions is
        /// quadr[0] + quadr[1] * 2 + quadr[1] * 4
        const quadrs_pos: [n_children][dim]i8 = switch (dim) {
            2 => .{ [_]i8{ 0, 0 }, [_]i8{ 1, 0 }, [_]i8{ 0, 1 }, [_]i8{ 1, 1 } },
            3 => .{ [_]i8{ 0, 0, 0 }, [_]i8{ 1, 0, 0 }, [_]i8{ 0, 1, 0 }, [_]i8{ 1, 1, 0 }, [_]i8{ 0, 0, 1 }, [_]i8{ 1, 0, 1 }, [_]i8{ 0, 1, 1 }, [_]i8{ 1, 1, 1 } },
            else => unreachable,
        };

        pub fn init() Self {
            var inst: Self = .{ .data = undefined, .children = undefined, .adj_neighbours = undefined, .parent = null };
            for (0..dim) |d| {
                inst.adj_neighbours[d][0] = null;
                inst.adj_neighbours[d][1] = null;
            }
            for (0..n_children) |n| {
                inst.children[n] = null;
            }
            return inst;
        }

        pub fn lvl(self: *const Self) i32 {
            var b: *const Self = self;
            var blvl: i32 = 0;
            while (b.parent != null) {
                blvl += 1;
                b = b.parent.?;
            }
            return blvl;
        }

        pub fn size_lvl(lvl_use: i32) f32 {
            var s_lvl: f32 = 1;
            for (0..@intCast(lvl_use)) |_| {
                s_lvl *= 0.5;
            }
            return s_lvl;
        }

        pub fn ratio_size_lvl(lvl_use: i32) i32 {
            var rs: i32 = 1;
            for (0..@intCast(lvl_use)) |_| {
                rs *= 2;
            }
            return rs;
        }

        pub fn size(self: *const Self) f32 {
            return Self.size_lvl(self.lvl());
        }

        pub inline fn is_leaf(self: *const Self) bool {
            return self.children[0] == null;
        }

        pub inline fn is_runned(self: *const Self) bool {
            if (self.is_leaf()) {
                return true;
            }
            for (self.children) |c| {
                if (c.?.is_leaf()) {
                    return true;
                }
            }
            return false;
        }

        pub fn neighbour_diff_lvl(self: *const Self, direction: [dim]i8) !?*Self {
            const nb = try self.neighbour(direction);
            if (nb != null) {
                return nb;
            }
            if (self.lvl() == 0) {
                return null;
            }
            const b_quadr = self.block_quadr().?;
            var dir_parent: [dim]i8 = undefined;
            for (0..dim) |d| {
                dir_parent[d] = @divFloor(b_quadr[d] + direction[d], 2);
            }
            return self.parent.?.neighbour(dir_parent);
        }

        pub fn neighbour(self: *const Self, direction: [dim]i8) !?*Self {
            // Number of non zero dimensions in direction
            var n_dim: i8 = 0;
            // Dimensions with non zero values. Init with invalid values
            var dim_walk: [dim]i8 = .{-100} ** dim;

            // Fill dim_walk and n_dir
            for (direction, 0..) |dir, i| {
                if (dir != 0 and dir != 1 and dir != -1) {
                    return BlockError.InvalidNeighbour;
                }
                if (dir != 0) {
                    dim_walk[@intCast(n_dim)] = @intCast(i);
                    n_dim += 1;
                }
            }
            if (n_dim == 0) {
                return BlockError.InvalidNeighbour;
            }
            if (n_dim == 1) {
                const nd = dim_walk[0];
                const dir = direction[@intCast(nd)];
                if (dir != -1 and dir != 1) {
                    return BlockError.InvalidNeighbour;
                }
                const p: usize = if (dir == -1) 0 else 1;
                return self.adj_neighbours[@intCast(nd)][p];
            }

            // For each dimension, try to resolve the neighbour recursively for higher dimensions
            const dir_slice = dim_walk[0..@intCast(n_dim)];

            for (dir_slice) |d_adj| {
                // 1D direction, direct neighbour to use
                var adj_dir: [dim]i8 = .{0} ** dim;
                // direction of rest of the neighbours, direct neighbour to use
                var rest_dir: [dim]i8 = .{0} ** dim;
                for (dir_slice) |d_check| {
                    if (d_adj != d_check) {
                        rest_dir[@intCast(d_check)] = direction[@intCast(d_check)];
                    } else {
                        adj_dir[@intCast(d_check)] = direction[@intCast(d_check)];
                    }
                }

                // Get adjacent neighbour and get the rest of neighbour from it
                const adj_nb = try self.neighbour(adj_dir);
                if (adj_nb != null) {
                    const nb_check = try adj_nb.?.neighbour(rest_dir);
                    if (nb_check != null) {
                        return nb_check;
                    }
                }
            }
            // No block was found, return null
            return null;
        }

        fn adj_neighbour_offsets(axis: u8, offset: i8) [dim]i8 {
            var full_offset: [dim]i8 = .{0} ** dim;
            for (0..dim) |d| {
                if (axis == d) {
                    full_offset[d] = offset;
                }
            }
            return full_offset;
        }

        fn update_children_neighbours(self: *Self) !void {
            for (self.children, Self.quadrs_pos) |c, q| {
                for (0..dim) |axis| {
                    const offsets: [2]i8 = .{ -1, 1 };
                    const offsets_idxs: [2]usize = .{ 0, 1 };
                    for (offsets, offsets_idxs) |offset, offset_idx| {
                        const nb_offset = adj_neighbour_offsets(@intCast(axis), offset);
                        // Full quadrant, in relation to this block (may be 2 or -1, for example)
                        var nb_full_quadr: [dim]i8 = .{0} ** dim;
                        // Quadrant of neighbour block
                        var nb_quadr: [dim]i8 = .{0} ** dim;
                        var nb_parent: [dim]i8 = .{0} ** dim;
                        for (0..dim) |d| {
                            nb_full_quadr[d] = q[d] + nb_offset[d];
                            nb_quadr[d] = @mod(nb_full_quadr[d], 2);
                            nb_parent[d] = @divFloor(nb_full_quadr[d], 2);
                        }
                        const parent: ?*Self = if (nb_parent[axis] == 0) self else try self.neighbour(nb_parent);
                        // Neighbour block doesn't exists or doesn't have children
                        if (parent == null or parent.?.is_leaf()) {
                            continue;
                        }
                        // Get neighbour from parent quadrant and update it
                        const nb = try parent.?.child_quadr(nb_quadr);
                        c.?.adj_neighbours[axis][offset_idx] = nb.?;
                        nb.?.adj_neighbours[axis][if (offset_idx == 0) 1 else 0] = c.?;
                    }
                }
            }
        }

        pub fn divide_single_block(self: *Self, children_use: [n_children]*Self) !void {
            try self.add_children(children_use);
            try self.update_children_neighbours();
        }

        pub fn add_children(self: *Self, children_add: [n_children]*Self) !void {
            for (children_add) |c| {
                if (c.parent != null) {
                    return BlockError.BlockHasParentAlready;
                }
            }
            if (!self.is_leaf()) {
                return BlockError.BlockHasChildrenAlready;
            }
            self.children = undefined;
            for (children_add) |c| {
                c.parent = self;
            }
            std.mem.copyForwards(?*Self, &(self.children), &children_add);
        }

        pub inline fn block_quadr(self: *const Self) ?[dim]i8 {
            const qi = self.block_quadr_idx();
            if (qi == null) {
                return null;
            }
            return Self.quadrs_pos[qi.?];
        }

        pub inline fn block_quadr_idx(self: *const Self) ?usize {
            if (self.parent == null) {
                return null;
            }
            const children = self.parent.?.children;

            const needle: [1]?*const Self = [1]?*const Self{self};
            // Use std.mem.indexOf to find the index of self in children
            const index = std.mem.indexOf(?*const Self, &children, &needle);
            if (index == null) {
                std.debug.panic("Couldn't find myself in parent. Why?", .{});
            }
            return index.?;
        }

        pub fn child_quadr_idx(self: *const Self, idx: usize) !?*Self {
            if (idx >= (1 << dim)) {
                return BlockError.QuadrNotExists;
            }
            if (self.is_leaf()) {
                return null;
            }
            return self.children[idx].?;
        }

        pub fn child_quadr(self: *const Self, quadr: [dim]i8) !?*Self {
            if (self.is_leaf()) {
                return null;
            }
            var idx: usize = 0;
            inline for (0..dim) |d| {
                if (quadr[d] != 0 and quadr[d] != 1) {
                    return BlockError.QuadrNotExists;
                }
                idx += @as(usize, @intCast(quadr[d])) * (1 << d);
            }
            return self.child_quadr_idx(idx);
        }
    };
}

fn list2ref(comptime T: type, objs: *[T.n_children]T) [T.n_children]*T {
    var refs: [T.n_children]*T = undefined;
    for (0..T.n_children) |i| {
        refs[i] = &(objs[i]);
    }
    return refs;
}

fn assert_block_properties(comptime dim: i32, T: type, block: *OctreeBlock(dim, T)) !void {
    const BlockType = OctreeBlock(dim, T);
    const n_children = BlockType.n_children;

    // Check children
    if (!block.is_leaf()) {
        for (0..n_children) |j| {
            const bc = block.children[j].?;
            try std.testing.expectEqual(block.lvl() + 1, bc.lvl());
            try std.testing.expectEqual(j, bc.block_quadr_idx());
            try std.testing.expectEqualSlices(
                i8,
                &BlockType.quadrs_pos[j],
                &bc.block_quadr().?,
            );
        }
    }
    const possible_nbs = all_possible_neighbours(dim);

    // All neighbours must have same level
    for (possible_nbs) |nb_dir| {
        const nb_ask = try block.neighbour(nb_dir);
        if (nb_ask == null) {
            continue;
        }
        const nb = nb_ask.?;
        // neighbours must have same level
        try std.testing.expectEqual(nb.lvl(), block.lvl());
    }
}

test "block simple_funcs" {
    inline for (.{ 2, 3 }) |dim| {
        const Block = OctreeBlock(dim, i32);
        const n_children = Block.n_children;

        // parent block creation and expects
        var b_parent = Block.init();

        try std.testing.expectEqual(0, b_parent.lvl());
        try std.testing.expectEqual(true, b_parent.is_runned());
        try std.testing.expectEqual(true, b_parent.is_leaf());

        var children: [n_children]Block = undefined;
        for (0..n_children) |i| {
            const b_child = Block.init();
            children[i] = b_child;
        }

        // Add children for testing
        try b_parent.add_children(list2ref(Block, &children));

        // Assert parent
        try std.testing.expectEqual(0, b_parent.lvl());
        try std.testing.expectEqual(true, b_parent.is_runned());
        try std.testing.expectEqual(false, b_parent.is_leaf());

        // Assert children
        try std.testing.expect(!b_parent.is_leaf());

        for (0..n_children) |i| {
            const c = b_parent.children[i].?;
            try std.testing.expectEqual(true, c.is_leaf());
            try std.testing.expectEqual(1, c.lvl());
            try std.testing.expectEqual(true, c.is_leaf());
            try std.testing.expectEqual(true, c.is_runned());
        }
    }
}

test "block quadrants and run" {
    inline for (.{ 2, 3 }) |dim| {
        const Block = OctreeBlock(dim, i32);
        const n_children = Block.n_children;

        // parent block creation and expects
        var b_parent: Block = Block.init();

        // Children are level 1 blocks
        var children: [n_children]Block = undefined;
        for (0..n_children) |i| {
            const b_child = Block.init();
            children[i] = b_child;
        }
        try b_parent.add_children(list2ref(Block, &children));

        for (0..n_children) |i| {
            try assert_block_properties(dim, i32, &children[i]);
        }

        // Now create grandchildrens (level 2 blocks)
        var grandchildren: [n_children][n_children]Block = undefined;
        for (0..n_children) |i| {
            for (0..n_children) |j| {
                const b_child = Block.init();
                grandchildren[i][j] = b_child;
            }
            // There is a children with no child yet, so parent must be runned
            try std.testing.expectEqual(b_parent.is_runned(), true);

            try children[i].add_children(list2ref(Block, &grandchildren[i]));

            // Check grandcihldrens
            for (0..n_children) |j| {
                try assert_block_properties(dim, i32, &grandchildren[i][j]);
            }
        }

        // At the end, with all grandchildren, parent must not be runned...
        try std.testing.expectEqual(false, b_parent.is_runned());
        for (children) |c| {
            try std.testing.expectEqual(true, c.is_runned());
            try std.testing.expectEqual(1, c.lvl());
        }
    }
}

test "block neighbours manual" {
    const dim = 3;
    const Block = OctreeBlock(dim, i32);

    var bzzz = Block.init();
    var bpzz = Block.init();
    var bpmz = Block.init();
    var bpmm = Block.init();
    var bzzm = Block.init();
    var bpzm = Block.init();

    bzzz.adj_neighbours[0][1] = &bpzz;
    bpzz.adj_neighbours[1][0] = &bpmz;
    bpmz.adj_neighbours[2][0] = &bpmm;
    bzzz.adj_neighbours[2][0] = &bzzm;
    bzzm.adj_neighbours[0][1] = &bpzm;

    try std.testing.expectEqual(&bpzz, bzzz.neighbour(.{ 1, 0, 0 }));
    try std.testing.expectEqual(&bpmz, bzzz.neighbour(.{ 1, -1, 0 }));
    try std.testing.expectEqual(&bpmm, bzzz.neighbour(.{ 1, -1, -1 }));

    try std.testing.expectEqual(&bzzm, bzzz.neighbour(.{ 0, 0, -1 }));
    try std.testing.expectEqual(&bpzm, bzzz.neighbour(.{ 1, 0, -1 }));
}

/// Forest of blocks using an Octree structure.
/// All octree blocks management and updates must be done through this interface
pub fn OctreeForest(comptime octree_dim: i32) type {
    return struct {
        const Self = @This();
        pub const dim = octree_dim;

        pub const BlockData = extern struct {
            idx: usize,
            start_ptr: usize,
            pos: [dim]f32,
        };

        pub const BlockType = OctreeBlock(dim, BlockData);

        allocator: std.mem.Allocator,
        /// All blocks in Forest
        all_blocks: std.ArrayList(BlockType),
        /// Level zero blocks, useful for finding blocks by position
        domain_blocks: []BlockType,
        /// Domain size (in number of blocks)
        domain_size: [dim]usize,
        /// Wheter each dimension is periodic or not
        domain_periodic: [dim]bool,

        /// Check if block must communicate in given direction
        pub fn block_communicate_direction(self: Self, block: *const BlockType, direction: [dim]i8) bool {
            const b_pos = block.data.pos;
            const block_size = block.size();
            for (0..dim) |d| {
                if (self.domain_periodic[d]) {
                    continue;
                }
                if (b_pos[d] == 0 and direction[d] == -1) {
                    return false;
                }
                if ((b_pos[d] + block_size) == @as(f32, @floatFromInt(self.domain_size[d])) and direction[d] == 1) {
                    return false;
                }
            }
            return true;
        }

        fn block_pos_min(self: *const Self, block: *const BlockType) [dim]f32 {
            const b_lvl = block.lvl();
            if (b_lvl == 0) {
                const b_idx = block.data.idx;
                const b_pos = self.domain_idx2pos(b_idx);
                var f_pos: [dim]f32 = undefined;
                for (0..dim) |d| {
                    f_pos[d] = @floatFromInt(b_pos[d]);
                }
                return f_pos;
            }
            const parent_pos = self.block_pos_min(block.parent.?);
            const block_size = block.size();
            var b_pos: [dim]f32 = undefined;
            const quadr = block.block_quadr().?;
            for (0..dim) |d| {
                var q: f32 = @floatFromInt(quadr[d]);
                q *= block_size;
                b_pos[d] = parent_pos[d] + q;
            }
            return b_pos;
        }

        const PosFind = union(enum) { abs_pos: [dim]f32, lvl_pos: struct { pos_quadr: [dim]i32, lvl: i8 } };

        pub fn find_block(self: *const Self, pos: PosFind, max_lvl: ?i8) ?*BlockType {
            _ = switch (pos) {
                .lvl_pos => |p_lvl| {
                    var abs_pos: [dim]f32 = undefined;
                    const b_size = BlockType.ratio_size_lvl(p_lvl.lvl);
                    for (0..dim) |d| {
                        abs_pos[d] = @floatFromInt(p_lvl.pos_quadr[d] * b_size);
                    }
                    const b = self.find_block(.{ .abs_pos = abs_pos }, max_lvl);
                    return b;
                },
                else => {},
            };

            const abs_pos: [dim]f32 = switch (pos) {
                .abs_pos => |p| blk: {
                    var p_use: [dim]f32 = p;
                    for (0..dim) |d| {
                        // Make an offset in border to allow for the max position to be the last block
                        if (p_use[d] == @as(f32, @floatFromInt(self.domain_size[d]))) {
                            p_use[d] -= 1e-6;
                        }
                    }
                    break :blk p_use;
                },
                else => unreachable,
            };

            var pos_lvl0: [dim]i32 = undefined;
            var pos_lvl0_u: [dim]u32 = undefined;
            for (0..dim) |d| {
                pos_lvl0[d] = @intFromFloat(@floor(abs_pos[d]));
                if (pos_lvl0[d] < 0 or pos_lvl0[d] > self.domain_size[d]) {
                    return null;
                }
                pos_lvl0_u[d] = @intCast(pos_lvl0[d]);
            }

            const block_lvl_0 = self.domain_block_from_pos_idx(pos_lvl0_u);
            if (block_lvl_0 == null) {
                return null;
            }

            var block_use = block_lvl_0.?;
            var b_lvl: i32 = 0;

            while (!block_use.is_leaf()) {
                const lvl_size = BlockType.size_lvl(b_lvl);
                const lvl_ratio: f32 = @floatFromInt(BlockType.ratio_size_lvl(b_lvl));
                b_lvl += 1;

                var quadr_use: [dim]i8 = undefined;
                for (0..dim) |d| {
                    const p_mod: f32 = @rem(abs_pos[d], lvl_size) * lvl_ratio;
                    quadr_use[d] = if (p_mod >= 0.5) 1 else 0;
                }
                const b_child = block_use.child_quadr(quadr_use) catch {
                    std.debug.panic("wrong, I nevear meant to be here...\n", .{});
                };
                block_use = b_child.?;

                // if (max_lvl != null) {
                //     if (block_use.lvl() >= max_lvl.?) {
                //         return block_use;
                //     }
                // }
            }
            std.debug.assert(block_use.is_leaf());
            return block_use;
        }

        pub inline fn domain_n_blocks(self: *const Self) usize {
            var n_blocks: usize = 1;
            for (self.domain_size) |n_b| {
                n_blocks *= n_b;
            }
            return n_blocks;
        }

        pub inline fn domain_pos2idx(self: *const Self, pos: [dim]u32) usize {
            var idx: usize = 0;
            var d_mul: usize = 1;
            for (0..dim) |d| {
                idx += pos[d] * d_mul;
                d_mul *= self.domain_size[d];
            }
            return idx;
        }

        pub inline fn domain_idx2pos(self: *const Self, idx: usize) [dim]u32 {
            const ds = self.domain_size;
            if (dim == 2) {
                return .{ @intCast(idx % ds[0]), @intCast(idx / ds[0]) };
            } else {
                return .{ @intCast(idx % ds[0]), @intCast((idx / ds[0]) % ds[1]), @intCast(idx / (ds[0] * ds[1])) };
            }
        }

        pub inline fn domain_block_from_pos_idx(self: *const Self, idx_pos: [dim]u32) ?*BlockType {
            const idx = self.domain_pos2idx(idx_pos);
            if (idx >= self.domain_n_blocks()) {
                return null;
            }
            return self.block_from_idx(idx);
        }

        pub fn init(allocator: std.mem.Allocator, domain_size: [dim]usize, domain_periodic: [dim]bool) Self {
            const self: Self = .{
                .allocator = allocator,
                .all_blocks = undefined,
                .domain_blocks = undefined,
                .domain_size = domain_size,
                .domain_periodic = domain_periodic,
            };
            return self;
        }

        pub fn initialize_blocks(self: *Self) !void {
            const n_blocks = self.domain_n_blocks();

            self.all_blocks = std.ArrayList(BlockType).init(self.allocator);
            try self.all_blocks.ensureTotalCapacity(n_blocks * 10);
            self.all_blocks.appendNTimesAssumeCapacity(BlockType.init(), n_blocks);
            self.domain_blocks = self.all_blocks.items[0..n_blocks];
            self.initialize_domain_blocks_neighbours();
        }

        fn initialize_domain_blocks_neighbours(self: *Self) void {
            const n_blocks = self.domain_n_blocks();
            const ds = self.domain_size;
            for (0..n_blocks) |idx| {
                const pos = self.domain_idx2pos(idx);
                const b = self.block_from_idx(idx);
                b.data.idx = idx;
                b.data.start_ptr = @intFromPtr(self.all_blocks.items.ptr);
                for (0..dim) |d| {
                    b.data.pos[d] = @floatFromInt(pos[d]);
                    const offset = 1;
                    var pos_offset = pos;
                    if (self.domain_periodic[d]) {
                        // This will always be positive, no need to check
                        pos_offset[d] = @intCast(@mod(offset + pos[d], ds[d]));
                    } else if ((pos[d] + offset) > 0 and (pos[d] + offset) < ds[d]) {
                        pos_offset[d] = @intCast(pos[d] + offset);
                    } else {
                        // Position outside domain
                        continue;
                    }
                    const idx_offset = self.domain_pos2idx(pos_offset);
                    std.debug.assert(idx_offset <= n_blocks); // internal error
                    const b_neighbour = &(self.all_blocks.items[idx_offset]);
                    b.adj_neighbours[d][1] = b_neighbour;
                    b_neighbour.adj_neighbours[d][0] = b;
                }
            }
        }

        pub fn free(self: *Self) void {
            self.all_blocks.deinit();
            self.domain_blocks = &.{};
        }

        fn request_new_blocks(self: *Self, comptime n_blocks_add: usize, blocks_ptrs: ?*[n_blocks_add]*BlockType) !void {
            const n_blocks_curr = self.all_blocks.items.len;
            const n_blocks_after = n_blocks_curr + n_blocks_add;

            const old_base_ptr: usize = @intFromPtr(self.all_blocks.items.ptr);

            for (0..n_blocks_add) |_| {
                var b = BlockType.init();
                b.data.start_ptr = @intFromPtr(self.all_blocks.items.ptr);
                b.data.idx = self.all_blocks.items.len;
                try self.all_blocks.append(b);
            }

            const new_base_ptr: usize = @intFromPtr(self.all_blocks.items.ptr);
            if (old_base_ptr != new_base_ptr) {
                self.update_blocks_pointers(old_base_ptr, new_base_ptr);
            }

            if (blocks_ptrs != null) {
                for (n_blocks_curr..n_blocks_after, 0..) |full_idx, idx| {
                    blocks_ptrs.?.*[idx] = self.block_from_idx(full_idx);
                }
            }
        }

        pub inline fn block_from_idx(self: *const Self, idx: usize) *BlockType {
            return &self.all_blocks.items[idx];
        }

        fn blocks_list_must_refine(
            self: *const Self,
            block_idx: usize,
            list_idxs: *std.ArrayList(usize),
            force_neighbour_refinement: bool,
        ) !void {
            // Divide neighbour blocks of higher level
            const possible_neighbours = all_possible_neighbours(dim);
            const block_check = self.block_from_idx(block_idx);

            // If a block is runned, but it's not a leaf, all its possible neighbours must be refined
            // So if this is not satisfied, we can just return
            // Or in case of forcing refinement
            if (!(block_check.is_runned() and !block_check.is_leaf()) and !force_neighbour_refinement) {
                return;
            }

            // Enforce that the neighbour blocks have the same level
            for (possible_neighbours) |offset_nb| {
                if (self.block_communicate_direction(block_check, offset_nb)) {
                    const nb = block_check.neighbour_diff_lvl(offset_nb) catch {
                        unreachable;
                    };
                    if (force_neighbour_refinement and nb == null) {
                        continue;
                    }
                    if (nb == null) {
                        std.debug.panic("This should never happen, why i don't have a neighbour available?\n", .{});
                    }
                    if (nb.?.lvl() == block_check.lvl()) {
                        continue;
                    }

                    const nb_idx = nb.?.data.idx;
                    const needle: [1]usize = .{nb_idx};
                    // Use std.mem.indexOf to find the index of self in children
                    const is_in_list = std.mem.indexOf(usize, list_idxs.items, &needle) != null;
                    if (!is_in_list) {
                        try list_idxs.append(nb_idx);
                        try self.blocks_list_must_refine(nb_idx, list_idxs, false);
                    }
                }
            }
        }

        fn divide_block_single(self: *Self, block_idx: usize) !void {
            const b_ref = self.block_from_idx(block_idx);
            if (!b_ref.is_leaf()) {
                return BlockError.BlockHasChildrenAlready;
            }
            var children_use: [BlockType.n_children]*BlockType = undefined;
            try self.request_new_blocks(BlockType.n_children, &children_use);
            try self.block_from_idx(block_idx).divide_single_block(children_use);

            for (children_use) |c| {
                c.data.pos = self.block_pos_min(c);
                std.debug.assert(c.is_leaf());
            }
            std.debug.assert(!self.block_from_idx(block_idx).is_leaf());
        }

        fn divide_block_recursive(
            self: *Self,
            block_idx: usize,
            list_idxs: *std.ArrayList(usize),
        ) !void {
            // Divide neighbour blocks of higher level before dividing myself
            const possible_neighbours = all_possible_neighbours(dim);

            const needle: [1]usize = .{block_idx};
            // Use std.mem.indexOf to find the index of self in children
            const idx_list = std.mem.indexOf(usize, list_idxs.items, &needle);
            // Block was already refined, or is in the list
            if (idx_list != null) {
                return;
            }
            const idx_added = list_idxs.items.len;
            try list_idxs.append(block_idx);
            defer _ = list_idxs.orderedRemove(idx_added);

            // Enforce that the neighbour blocks have the same level
            for (possible_neighbours) |offset_nb| {
                const block_check = self.block_from_idx(block_idx);
                if (!self.block_communicate_direction(block_check, offset_nb)) {
                    continue;
                }
                const nb = block_check.neighbour_diff_lvl(offset_nb) catch {
                    unreachable;
                };
                if (nb == null) {
                    std.debug.panic("This should never happen, why i don't have a neighbour available?\n", .{});
                }
                if (nb.?.lvl() < block_check.lvl() and nb.?.is_leaf()) {
                    try self.divide_block_recursive(nb.?.data.idx, list_idxs);
                }
            }

            try self.divide_block_single(block_idx);
        }

        /// Divide a block in half in its dimension, generating 2^dim new ones.
        ///
        /// IMPORTANT: this function may invalidate previous pointers to blocks, so make sure to
        /// update previous block pointers to get new valid ones
        pub fn divide_block(self: *Self, block_idx: usize) !void {
            var idxs_to_refine = std.ArrayList(usize).init(self.allocator);
            defer idxs_to_refine.deinit();
            try self.divide_block_recursive(block_idx, &idxs_to_refine);
        }

        fn assert_block_neighbours(self: *const Self) void {
            const possible_neighbours = all_possible_neighbours(dim);
            // Enforce that the neighbour blocks have at most 1 level of difference
            // (requires update using parent block)
            for (0..self.all_blocks.items.len) |idx| {
                const block = self.block_from_idx(idx);
                for (possible_neighbours) |offset_nb| {
                    const nb = block.neighbour(offset_nb) catch {
                        std.debug.panic("why do I have an invalid neighbour?\n", .{});
                    };
                    const communicate = self.block_communicate_direction(block, offset_nb);
                    if (communicate and !block.is_leaf() and block.is_runned()) {
                        std.debug.assert(nb != null);
                    }
                }
            }
        }

        fn get_new_pointer(base_old: usize, base_new: usize, old_ptr: usize) usize {
            if (old_ptr < base_old) {
                std.debug.panic("Why i'm sending old pts less that old base? old ptr {} base old {}", .{ old_ptr, base_old });
            }
            const offset = old_ptr - base_old;
            return base_new + offset;
        }

        /// When new blocks are added, the old ones, may require pointer update due to new allocations
        /// This function updates the pointers inside all blocks for the new references, including
        /// parent, children and neighbour.
        /// All pointers used before becomes invalid
        fn update_blocks_pointers(self: Self, old_base_ptr: usize, new_base_ptr: usize) void {
            for (0..self.all_blocks.items.len) |idx| {
                const b = &(self.all_blocks.items[idx]);
                b.data.start_ptr = @intFromPtr(self.all_blocks.items.ptr);
                b.data.idx = idx;
                if (b.parent != null) {
                    b.parent = @ptrFromInt(Self.get_new_pointer(
                        old_base_ptr,
                        new_base_ptr,
                        @intFromPtr(b.parent.?),
                    ));
                }
                if (!b.is_leaf()) {
                    for (0..BlockType.n_children) |cidx| {
                        b.children[cidx] = @ptrFromInt(Self.get_new_pointer(
                            old_base_ptr,
                            new_base_ptr,
                            @intFromPtr(b.children[cidx]),
                        ));
                    }
                }
                for (0..dim) |d| {
                    for (0..2) |nb_idx| {
                        const nb = b.adj_neighbours[d][nb_idx];
                        if (nb == null) {
                            continue;
                        }
                        b.adj_neighbours[d][nb_idx] = @ptrFromInt(get_new_pointer(
                            old_base_ptr,
                            new_base_ptr,
                            @intFromPtr(nb.?),
                        ));
                    }
                }
            }
        }
    };
}

fn assert_octree_neighbours_properties(
    comptime dim: i32,
    forest: *OctreeForest(dim),
    block: *OctreeForest(dim).BlockType,
) !void {
    // Check that neighbour parent blocks have at most one level of difference
    const possible_nbs = all_possible_neighbours(dim);
    var b_quadr: [dim]i8 = .{0} ** dim;
    if (block.lvl() > 0) {
        b_quadr = block.block_quadr().?;
    }

    for (possible_nbs) |nb_dir| {
        const communicate = forest.block_communicate_direction(block, nb_dir);
        const nb = try block.neighbour_diff_lvl(nb_dir);
        const dn = try block.neighbour(nb_dir);
        if (dn != null and nb != null) {
            try std.testing.expectEqual(nb, dn);
        } else if (dn == null and nb != null) {
            try std.testing.expectEqual(block.lvl() - 1, nb.?.lvl());
        }
        if (communicate) {
            // If a block must communicate and is refined, all its neighbours must be refined
            // This is to prevent neighbour blocks with more than one level of difference
            try std.testing.expect(nb != null);
            try std.testing.expect(nb.?.lvl() <= block.lvl());
            if (nb.?.lvl() < block.lvl()) {
                try std.testing.expect(nb.?.lvl() == block.lvl() - 1);
                try std.testing.expect(nb.?.is_leaf());
            }
        } else {
            try std.testing.expect(dn == null);
            try std.testing.expect(nb == null);
        }
    }
}

fn assert_octree_domain(comptime dim: i32, forest: OctreeForest(dim), pos: [dim]u32, offset: [dim]i8) !void {
    const ds = forest.domain_size;
    const pd = forest.domain_periodic;
    var p_final: [dim]i32 = undefined;
    var p_final_u: [dim]u32 = undefined;
    var is_outside: bool = false;
    for (0..dim) |d| {
        p_final[d] = if (pd[d]) @as(i32, @intCast(@mod(@as(i32, @intCast(pos[d])) + @as(i32, offset[d]), @as(i32, @intCast(ds[d]))))) else @as(i32, @intCast(pos[d])) + @as(i32, offset[d]);
        if (p_final[d] < 0 or p_final[d] >= ds[d]) {
            is_outside = true;
        } else {
            p_final_u[d] = @intCast(p_final[d]);
        }
    }
    const b = forest.domain_block_from_pos_idx(pos).?;
    const nb = try b.neighbour(offset);
    if (is_outside) {
        try std.testing.expectEqual(null, nb);
    } else {
        const nb_domain = forest.domain_block_from_pos_idx(p_final_u);
        try std.testing.expectEqual(nb_domain.?, nb.?);
    }
}

fn assert_octree_idxs(comptime dim: i32, forest: OctreeForest(dim)) !void {
    for (0..forest.all_blocks.items.len) |idx| {
        const b = &forest.all_blocks.items[idx];
        const idx_func = b.data.idx;
        try std.testing.expectEqual(idx, idx_func);
    }
}

test "domain_initialization" {
    const allocator = std.testing.allocator;

    inline for (.{2}) |dim| {
        const Forest = OctreeForest(dim);
        const domain_size: [dim]usize = if (dim == 3) .{ 12, 18, 13 } else .{ 12, 18 };
        const ds = domain_size;
        const periodic_dim: [dim]bool = if (dim == 3) .{ true, false, true } else .{ true, false };
        var forest = Forest.init(allocator, domain_size, periodic_dim);
        defer forest.free();

        try std.testing.expectEqual(if (dim == 2) ds[0] * ds[1] else ds[0] * ds[1] * ds[2], forest.domain_n_blocks());
        try forest.initialize_blocks();

        try assert_octree_idxs(dim, forest);
        for (0..forest.domain_n_blocks()) |idx| {
            const b = &forest.all_blocks.items[idx];
            try assert_octree_neighbours_properties(dim, &forest, b);
            const pos = forest.domain_idx2pos(idx);
            for (0..dim) |d| {
                {
                    const nb0 = b.adj_neighbours[d][0];
                    if (nb0 == null) {
                        try std.testing.expect(pos[1] == 0);
                    } else {
                        const nb_neg = nb0.?.adj_neighbours[d][1].?;
                        try std.testing.expectEqual(b, nb_neg);
                    }
                }
                {
                    const nb1 = b.adj_neighbours[d][1];
                    if (nb1 == null) {
                        try std.testing.expect(pos[1] == domain_size[1] - 1);
                    } else {
                        const nb_neg = nb1.?.adj_neighbours[d][0].?;
                        try std.testing.expectEqual(b, nb_neg);
                    }
                }
            }
        }

        const nt = 4;
        const pos_orig: [nt][3]u32 = .{ [_]u32{ 0, 0, 0 }, [_]u32{ 5, 6, 7 }, [_]u32{ 4, domain_size[1] - 1, 3 }, [_]u32{ domain_size[0] - 1, 0, 3 } };
        const offsets = all_possible_neighbours(dim);

        for (pos_orig) |p| {
            for (offsets) |o| {
                var pos: [dim]u32 = undefined;
                var odim: [dim]i8 = undefined;
                for (0..dim) |d| {
                    odim[d] = o[d];
                    pos[d] = p[d];
                }
                try assert_octree_domain(dim, forest, pos, odim);
            }
        }
    }
}

test "domain_block_division" {
    const allocator = std.testing.allocator;

    inline for (.{ 2, 3 }) |dim| {
        const Forest = OctreeForest(dim);
        const domain_size: [dim]usize = if (dim == 3) .{ 12, 6, 8 } else .{ 2, 2 };
        const periodic_dim: [dim]bool = if (dim == 3) .{ true, true, true } else .{ true, true };
        var forest = Forest.init(allocator, domain_size, periodic_dim);
        defer forest.free();
        try forest.initialize_blocks();

        for (0..5) |_| {
            const last_idx: usize = forest.all_blocks.items.len - 1;
            try forest.divide_block(last_idx);
            try std.testing.expectEqual(false, forest.block_from_idx(last_idx).is_leaf());
            try assert_octree_idxs(dim, forest);
            for (0..forest.all_blocks.items.len) |idx| {
                const b = &forest.all_blocks.items[idx];
                try assert_block_properties(dim, OctreeForest(dim).BlockData, b);
                try assert_octree_neighbours_properties(dim, &forest, b);
            }
        }
    }
}

test "domain_find_block" {
    const allocator = std.testing.allocator;

    inline for (.{ 2, 3 }) |dim| {
        const Forest = OctreeForest(dim);
        const ds: [dim]usize = if (dim == 3) .{ 12, 18, 7 } else .{ 12, 18 };
        const periodic_dim: [dim]bool = if (dim == 3) .{ true, false, true } else .{ true, false };
        var forest = Forest.init(allocator, ds, periodic_dim);
        defer forest.free();
        try forest.initialize_blocks();

        const pos_valid: [dim]f32 = if (dim == 3) .{ ds[0], 0, 4 } else .{ ds[0], 0 };
        const pos_invalid: [dim]f32 = if (dim == 3) .{ ds[0] + 1, 2, ds[2] - 1 } else .{ 0, -1 };

        for (0..10) |_| {
            const last_idx: usize = forest.all_blocks.items.len - 1;
            const lvl_expect = forest.block_from_idx(last_idx).lvl() + 1;
            try forest.divide_block(last_idx);
            const b_valid = forest.find_block(.{ .abs_pos = pos_valid }, null);
            try std.testing.expect(b_valid != null);

            const b = forest.block_from_idx(last_idx);
            const b_last = forest.find_block(.{ .abs_pos = b.data.pos }, null);
            try std.testing.expect(b_last != null);
            try std.testing.expectEqual(lvl_expect, b_last.?.lvl());
            try std.testing.expectEqual(false, b.is_leaf());
            try std.testing.expectEqual(b.lvl() + 1, b_last.?.lvl());

            const b_invalid = forest.find_block(.{ .abs_pos = pos_invalid }, null);
            try std.testing.expect(b_invalid == null);
        }
    }
}

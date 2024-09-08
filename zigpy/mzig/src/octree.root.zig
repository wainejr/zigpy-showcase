const std = @import("std");
const testing = std.testing;
const octree = @import("multiblock/octree.zig");

var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
const allocator = arena.allocator();

fn arr2struct(comptime dim: usize, comptime T: type) type {
    if (dim != 2 and dim != 3) {
        std.debug.panic("Only 2 or 3 dimensions are possible", .{});
    }
    if (dim == 2) {
        return extern struct {
            const Self = @This();
            x: T,
            y: T,
            pub fn new(val: [dim]T) Self {
                return .{ .x = val[0], .y = val[1] };
            }
            pub fn arr(self: Self) [dim]T {
                return .{ self.x, self.y };
            }
        };
    }
    return extern struct {
        const Self = @This();
        x: T,
        y: T,
        z: T,
        pub fn new(val: [dim]T) Self {
            return .{ .x = val[0], .y = val[1], .z = val[2] };
        }
        pub fn arr(self: Self) [dim]T {
            return .{ self.x, self.y, self.z };
        }
    };
}
const a2s = arr2struct;

fn BlockCalls(comptime dim: i32) type {
    return struct {
        pub const BlockType = octree.OctreeForest(dim).BlockType;

        pub fn lvl(self: [*c]BlockType) callconv(.C) i32 {
            return self.*.lvl();
        }
        pub fn size(self: [*c]BlockType) callconv(.C) f32 {
            return self.*.size();
        }
        pub fn is_leaf(self: [*c]BlockType) callconv(.C) bool {
            return self.*.is_leaf();
        }
        pub fn is_runned(self: [*c]BlockType) callconv(.C) bool {
            return self.*.is_runned();
        }
        pub fn neighbour_diff_lvl(self: [*c]BlockType, direction: a2s(dim, i8)) callconv(.C) [*c]BlockType {
            return self.*.neighbour_diff_lvl(direction.arr()) catch {
                std.debug.panic("Error on neighbour diff lvl\n", .{});
            };
        }
        pub fn neighbour(self: [*c]BlockType, direction: a2s(dim, i8)) callconv(.C) [*c]BlockType {
            return self.*.neighbour(direction.arr()) catch |err| {
                std.debug.panic("Error on neighbour {any}\n", .{err});
            };
        }
        pub fn block_quadr(self: [*c]BlockType) callconv(.C) a2s(dim, i8) {
            const res = self.*.block_quadr();
            return a2s(dim, i8).new(res);
        }
        pub fn block_quadr_idx(self: [*c]BlockType) callconv(.C) usize {
            const res = self.*.block_quadr_idx();
            return res;
        }
        pub fn child_quadr_idx(self: [*c]BlockType, idx: usize) callconv(.C) [*c]BlockType {
            const res = self.*.child_quadr_idx(idx) catch |err| {
                std.debug.panic("Error on child quadr idx {any}\n", .{err});
            };
            return res;
        }
        pub fn child_quadr(self: [*c]BlockType, quadr: a2s(dim, i8)) callconv(.C) [*c]BlockType {
            const res = self.*.child_quadr(quadr.arr()) catch |err| {
                std.debug.panic("Error on child quadr {any}\n", .{err});
            };
            return res;
        }
    };
}

comptime {
    @export(BlockCalls(2).lvl, .{ .name = "Block2D__lvl", .linkage = .strong });
    @export(BlockCalls(2).size, .{ .name = "Block2D__size", .linkage = .strong });
    @export(BlockCalls(2).is_leaf, .{ .name = "Block2D__is_leaf", .linkage = .strong });
    @export(BlockCalls(2).is_runned, .{ .name = "Block2D__is_runned", .linkage = .strong });
    @export(BlockCalls(2).neighbour_diff_lvl, .{ .name = "Block2D__neighbour_diff_lvl", .linkage = .strong });
    @export(BlockCalls(2).neighbour, .{ .name = "Block2D__neighbour", .linkage = .strong });
    @export(BlockCalls(2).block_quadr, .{ .name = "Block2D__block_quadr", .linkage = .strong });
    @export(BlockCalls(2).block_quadr_idx, .{ .name = "Block2D__block_quadr_idx", .linkage = .strong });
    @export(BlockCalls(2).child_quadr_idx, .{ .name = "Block2D__child_quadr_idx", .linkage = .strong });
    @export(BlockCalls(2).child_quadr, .{ .name = "Block2D__child_quadr", .linkage = .strong });

    @export(BlockCalls(3).lvl, .{ .name = "Block3D__lvl", .linkage = .strong });
    @export(BlockCalls(3).size, .{ .name = "Block3D__size", .linkage = .strong });
    @export(BlockCalls(3).is_leaf, .{ .name = "Block3D__is_leaf", .linkage = .strong });
    @export(BlockCalls(3).is_runned, .{ .name = "Block3D__is_runned", .linkage = .strong });
    @export(BlockCalls(3).neighbour_diff_lvl, .{ .name = "Block3D__neighbour_diff_lvl", .linkage = .strong });
    @export(BlockCalls(3).neighbour, .{ .name = "Block3D__neighbour", .linkage = .strong });
    @export(BlockCalls(3).block_quadr, .{ .name = "Block3D__block_quadr", .linkage = .strong });
    @export(BlockCalls(3).block_quadr_idx, .{ .name = "Block3D__block_quadr_idx", .linkage = .strong });
    @export(BlockCalls(3).child_quadr_idx, .{ .name = "Block3D__child_quadr_idx", .linkage = .strong });
    @export(BlockCalls(3).child_quadr, .{ .name = "Block3D__child_quadr", .linkage = .strong });
}

fn ForestCalls(comptime dim: i32) type {
    const ForestType = octree.OctreeForest(dim);
    const BlockType = ForestType.BlockType;
    return extern struct {
        const Self = @This();
        var forest: ?ForestType = null;

        forest_ptr: u64,
        dim: i32,

        pub fn new(domain_size: a2s(dim, usize), periodic_domain: a2s(dim, bool)) callconv(.C) Self {
            if (Self.forest != null) {
                std.debug.panic("I already have a forest, free me than create a new one\n", .{});
            }

            Self.forest = ForestType.init(domain_size.arr(), periodic_domain.arr());
            Self.forest.?.initialize_blocks(allocator) catch |err| {
                std.debug.panic("Error on new forest block initialization {any}\n", .{err});
            };

            // You can use this to return a index for the forest, in case more forest are allocated
            const c_forest: Self = .{
                .forest_ptr = 0,
                .dim = dim,
            };
            return c_forest;
        }

        pub fn free(self: *Self) callconv(.C) void {
            if (Self.forest == null) {
                std.debug.panic("I don't have a forest, why you're freeing me?\n", .{});
            }
            Self.forest.?.free();
            Self.forest = null;
            self.forest_ptr = 0;
        }

        pub fn block_from_idx(self: *const Self, idx: usize) callconv(.C) ?*BlockType {
            _ = self;
            const frt = Self.forest.?;
            return frt.block_from_idx(idx);
        }

        pub fn block_from_pos(self: *const Self, pos: a2s(dim, f32), max_lvl: i8) callconv(.C) ?*BlockType {
            _ = self;
            const frt = Self.forest.?;
            return frt.find_block(.{ .abs_pos = pos.arr() }, if (max_lvl >= 0) max_lvl else null);
        }

        pub fn block_from_quadr(self: *const Self, pos_quadr: a2s(dim, i32), quadr_lvl: i8, max_lvl: i8) callconv(.C) ?*BlockType {
            _ = self;
            const frt = Self.forest.?;
            return frt.find_block(.{
                .lvl_pos = .{
                    .pos_quadr = pos_quadr.arr(),
                    .lvl = quadr_lvl,
                },
            }, if (max_lvl >= 0) max_lvl else null);
        }

        pub fn divide_block(self: *const Self, block_idx: usize) callconv(.C) void {
            _ = self;
            var frt = Self.forest.?;
            frt.divide_block(allocator, block_idx) catch |err| {
                std.debug.panic("Error on dividing block {any}\n", .{err});
            };
        }
    };
}

comptime {
    @export(ForestCalls(2).new, .{ .name = "Forest2D__new", .linkage = .strong });
    @export(ForestCalls(2).free, .{ .name = "Forest2D__free", .linkage = .strong });
    @export(ForestCalls(2).block_from_idx, .{ .name = "Forest2D__block_from_idx", .linkage = .strong });
    @export(ForestCalls(2).block_from_pos, .{ .name = "Forest2D__block_from_pos", .linkage = .strong });
    @export(ForestCalls(2).block_from_quadr, .{ .name = "Forest2D__block_from_quadr", .linkage = .strong });
    @export(ForestCalls(2).divide_block, .{ .name = "Forest2D__divide_block", .linkage = .strong });

    @export(ForestCalls(3).new, .{ .name = "Forest3D__new", .linkage = .strong });
    @export(ForestCalls(3).free, .{ .name = "Forest3D__free", .linkage = .strong });
    @export(ForestCalls(3).block_from_idx, .{ .name = "Forest3D__block_from_idx", .linkage = .strong });
    @export(ForestCalls(3).block_from_pos, .{ .name = "Forest3D__block_from_pos", .linkage = .strong });
    @export(ForestCalls(3).block_from_quadr, .{ .name = "Forest3D__block_from_quadr", .linkage = .strong });
    @export(ForestCalls(3).divide_block, .{ .name = "Forest3D__divide_block", .linkage = .strong });
}

test "test octree simple oper" {
    // const typeInfo = @typeName(OctreeForest2D.BlockType);
    inline for (.{ 2, 3 }) |dim| {
        const domain_size = if (dim == 2) .{ 2, 5 } else .{ 2, 5, 4 };
        const periodic_dim = if (dim == 2) .{ true, false } else .{ true, false, true };
        var forest = ForestCalls(dim).new(
            a2s(dim, usize).new(domain_size),
            a2s(dim, bool).new(periodic_dim),
        );
        forest.free();
    }
}

export fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}

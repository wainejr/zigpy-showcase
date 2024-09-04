pub const VelSet = enum {
    D2Q9,
    D3Q15,
    D3Q19,
    D3Q27,

    pub fn dim(self: @This()) u8 {
        return switch (self) {
            .D2Q9 => 2,
            else => 3,
        };
    }
    pub fn n_pop(self: @This()) u8 {
        return switch (self) {
            .D2Q9 => 9,
            .D3Q15 => 15,
            .D3Q19 => 19,
            .D3Q27 => 27,
        };
    }
    pub fn dirs(comptime self: @This()) [self.n_pop()][self.dim()]i8 {
        const d: [self.n_pop()][self.dim()]i8 = switch (self) {
            .D2Q9 => .{
                [_]i8{ 0, 0 },
                [_]i8{ 1, 0 },
                [_]i8{ 0, 1 },
                [_]i8{ -1, 0 },
                [_]i8{ 0, -1 },
                [_]i8{ 1, 1 },
                [_]i8{ -1, 1 },
                [_]i8{ -1, -1 },
                [_]i8{ 1, -1 },
            },
            .D3Q15 => .{
                [_]i8{ 0, 0, 0 },
                [_]i8{ 1, 0, 0 },
                [_]i8{ -1, 0, 0 },
                [_]i8{ 0, 1, 0 },
                [_]i8{ 0, -1, 0 },
                [_]i8{ 0, 0, 1 },
                [_]i8{ 0, 0, -1 },
                [_]i8{ 1, 1, 1 },
                [_]i8{ -1, -1, -1 },
                [_]i8{ 1, 1, -1 },
                [_]i8{ -1, -1, 1 },
                [_]i8{ 1, -1, 1 },
                [_]i8{ -1, 1, -1 },
                [_]i8{ -1, 1, 1 },
                [_]i8{ 1, -1, -1 },
            },
            .D3Q19 => .{
                [_]i8{ 0, 0, 0 },
                [_]i8{ 1, 0, 0 },
                [_]i8{ -1, 0, 0 },
                [_]i8{ 0, 1, 0 },
                [_]i8{ 0, -1, 0 },
                [_]i8{ 0, 0, 1 },
                [_]i8{ 0, 0, -1 },
                [_]i8{ 1, 1, 0 },
                [_]i8{ -1, -1, 0 },
                [_]i8{ 1, 0, 1 },
                [_]i8{ -1, 0, -1 },
                [_]i8{ 0, 1, 1 },
                [_]i8{ 0, -1, -1 },
                [_]i8{ 1, -1, 0 },
                [_]i8{ -1, 1, 0 },
                [_]i8{ 1, 0, -1 },
                [_]i8{ -1, 0, 1 },
                [_]i8{ 0, 1, -1 },
                [_]i8{ 0, -1, 1 },
            },
            .D3Q27 => .{
                [_]i8{ 0, 0, 0 },
                [_]i8{ 1, 0, 0 },
                [_]i8{ -1, 0, 0 },
                [_]i8{ 0, 1, 0 },
                [_]i8{ 0, -1, 0 },
                [_]i8{ 0, 0, 1 },
                [_]i8{ 0, 0, -1 },
                [_]i8{ 1, 1, 0 },
                [_]i8{ -1, -1, 0 },
                [_]i8{ 1, 0, 1 },
                [_]i8{ -1, 0, -1 },
                [_]i8{ 0, 1, 1 },
                [_]i8{ 0, -1, -1 },
                [_]i8{ 1, -1, 0 },
                [_]i8{ -1, 1, 0 },
                [_]i8{ 1, 0, -1 },
                [_]i8{ -1, 0, 1 },
                [_]i8{ 0, 1, -1 },
                [_]i8{ 0, -1, 1 },
                [_]i8{ 1, 1, 1 },
                [_]i8{ -1, -1, -1 },
                [_]i8{ 1, 1, -1 },
                [_]i8{ -1, -1, 1 },
                [_]i8{ 1, -1, 1 },
                [_]i8{ -1, 1, -1 },
                [_]i8{ -1, 1, 1 },
                [_]i8{ 1, -1, -1 },
            },
        };
        return d;
    }
    pub const dirD2Q9 = VelSet.dirs(VelSet.D2Q9);
    pub const dirD3Q15 = VelSet.dirs(VelSet.D3Q15);
    pub const dirD3Q19 = VelSet.dirs(VelSet.D3Q19);
    pub const dirD3Q27 = VelSet.dirs(VelSet.D3Q27);
    pub const options = .{ VelSet.D2Q9, VelSet.D3Q15, VelSet.D3Q19, VelSet.D3Q27 };
};

test "vel set consistency" {
    const std = @import("std");
    const debug = std.debug;

    inline for (VelSet.options) |vs| {
        const dirs = VelSet.dirs(vs);
        const dim = VelSet.dim(vs);
        const n_pop = VelSet.n_pop(vs);
        debug.assert(dirs.len == n_pop);
        for (dirs) |dir| {
            debug.assert(dir.len == dim);
            var opp_vel: [3]i8 = .{0} ** 3;
            for (dir, 0..) |d, i| {
                opp_vel[i] = -d;
            }
            // Assert that opposite velocity is present
            for (dirs) |other_dir| {
                for (0..dim) |i| {
                    // In case it's different, break
                    if (other_dir[i] != dir[i]) {
                        break;
                    }
                } else {
                    // Get to final, then break the outer loop, all went ok
                    break;
                }
            } else {
                debug.panic("velocity {any} doesn't have opposite in velocity set", .{dir});
            }
        }
    }
}

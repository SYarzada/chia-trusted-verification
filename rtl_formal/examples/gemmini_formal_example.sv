module GemminiFormalExample (
    input  wire       clock,
    input  wire       reset,
    input  wire       allocated,
    input  wire       issued,
    input  wire [3:0] deps,
    input  wire       req_valid,
    input  wire       req_ready,
    input  wire       write_valid,
    input  wire       write_ready
);
    wire issue_valid = allocated && !issued && (deps == 4'b0000);
    wire req_fire = req_valid && req_ready;
    wire write_fire = write_valid && write_ready;

    // CHIA_FORMAL_COLLATERAL
endmodule

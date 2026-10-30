// CHIA-BEGIN-AGENT-COLLATERAL
always @(posedge clock) begin
    if (!reset) begin
        assert(!issue_valid || (allocated && !issued && deps == 4'b0000));
        cover(issue_valid && allocated && !issued && deps == 4'b0000);
    end
end
// CHIA-END-AGENT-COLLATERAL

	.text
	.file	"test1.c"
	.globl	test1                           # -- Begin function test1
	.p2align	4, 0x90
	.type	test1,@function
test1:                                  # @test1
	.cfi_startproc
# %bb.0:
	movq	%rdx, %rax
	subq	%rdi, %rax
	movq	%rdx, %rcx
	subq	%rsi, %rcx
	cmpq	$32, %rax
	setb	%r8b
	cmpq	$32, %rcx
	setb	%al
	orb	%r8b, %al
	xorl	%ecx, %ecx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_3:                                #   in Loop: Header=BB0_1 Depth=1
	incl	%ecx
	cmpl	$20000000, %ecx                 # imm = 0x1312D00
	je	.LBB0_4
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
                                        #     Child Loop BB0_5 Depth 2
	xorl	%r8d, %r8d
	testb	%al, %al
	je	.LBB0_2
	.p2align	4, 0x90
.LBB0_5:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%rdi,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	addss	(%rsi,%r8,4), %xmm0
	movss	%xmm0, (%rdx,%r8,4)
	movss	4(%rdi,%r8,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	4(%rsi,%r8,4), %xmm0
	movss	%xmm0, 4(%rdx,%r8,4)
	movss	8(%rdi,%r8,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	8(%rsi,%r8,4), %xmm0
	movss	%xmm0, 8(%rdx,%r8,4)
	movss	12(%rdi,%r8,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	addss	12(%rsi,%r8,4), %xmm0
	movss	%xmm0, 12(%rdx,%r8,4)
	addq	$4, %r8
	cmpq	$1024, %r8                      # imm = 0x400
	jne	.LBB0_5
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movups	(%rdi,%r8,4), %xmm0
	movups	16(%rdi,%r8,4), %xmm1
	movups	(%rsi,%r8,4), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%rsi,%r8,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%rdx,%r8,4)
	movups	%xmm0, 16(%rdx,%r8,4)
	movups	32(%rdi,%r8,4), %xmm0
	movups	48(%rdi,%r8,4), %xmm1
	movups	32(%rsi,%r8,4), %xmm2
	addps	%xmm0, %xmm2
	movups	48(%rsi,%r8,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, 32(%rdx,%r8,4)
	movups	%xmm0, 48(%rdx,%r8,4)
	addq	$16, %r8
	cmpq	$1024, %r8                      # imm = 0x400
	jne	.LBB0_2
	jmp	.LBB0_3
.LBB0_4:
	retq
.Lfunc_end0:
	.size	test1, .Lfunc_end0-test1
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

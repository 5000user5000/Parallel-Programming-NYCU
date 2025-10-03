	.text
	.file	"test2.c"
	.globl	test2                           # -- Begin function test2
	.p2align	4, 0x90
	.type	test2,@function
test2:                                  # @test2
	.cfi_startproc
# %bb.0:
	xorl	%eax, %eax
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_11:                               #   in Loop: Header=BB0_1 Depth=1
	incl	%eax
	cmpl	$20000000, %eax                 # imm = 0x1312D00
	je	.LBB0_12
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	xorl	%ecx, %ecx
	jmp	.LBB0_2
	.p2align	4, 0x90
.LBB0_10:                               #   in Loop: Header=BB0_2 Depth=2
	addq	$4, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	je	.LBB0_11
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	(%rdi,%rcx,4), %xmm1
	movups	%xmm1, (%rdx,%rcx,4)
	movaps	(%rsi,%rcx,4), %xmm0
	ucomiss	%xmm1, %xmm0
	ja	.LBB0_3
# %bb.4:                                #   in Loop: Header=BB0_2 Depth=2
	cmpltps	%xmm0, %xmm1
	pextrw	$2, %xmm1, %r8d
	testb	$1, %r8b
	jne	.LBB0_5
.LBB0_6:                                #   in Loop: Header=BB0_2 Depth=2
	pextrw	$4, %xmm1, %r8d
	testb	$1, %r8b
	jne	.LBB0_7
.LBB0_8:                                #   in Loop: Header=BB0_2 Depth=2
	pextrw	$6, %xmm1, %r8d
	testb	$1, %r8b
	je	.LBB0_10
	jmp	.LBB0_9
	.p2align	4, 0x90
.LBB0_3:                                #   in Loop: Header=BB0_2 Depth=2
	movss	%xmm0, (%rdx,%rcx,4)
	cmpltps	%xmm0, %xmm1
	pextrw	$2, %xmm1, %r8d
	testb	$1, %r8b
	je	.LBB0_6
.LBB0_5:                                #   in Loop: Header=BB0_2 Depth=2
	movaps	%xmm0, %xmm2
	shufps	$85, %xmm0, %xmm2               # xmm2 = xmm2[1,1],xmm0[1,1]
	movss	%xmm2, 4(%rdx,%rcx,4)
	pextrw	$4, %xmm1, %r8d
	testb	$1, %r8b
	je	.LBB0_8
.LBB0_7:                                #   in Loop: Header=BB0_2 Depth=2
	movaps	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	movss	%xmm2, 8(%rdx,%rcx,4)
	pextrw	$6, %xmm1, %r8d
	testb	$1, %r8b
	je	.LBB0_10
.LBB0_9:                                #   in Loop: Header=BB0_2 Depth=2
	shufps	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movss	%xmm0, 12(%rdx,%rcx,4)
	jmp	.LBB0_10
.LBB0_12:
	retq
.Lfunc_end0:
	.size	test2, .Lfunc_end0-test2
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

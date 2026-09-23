# Unmodified sampling-loop excerpt, TencentAI4S/HuDiff bb7636f182699f98c37855dad05a5c6c61b576bd,
# nanobody_scripts/sample_for_nano_cdr.py:174-213 (PolyForm Noncommercial 1.0.0).
# Executed only by the offline oracle test with injected model/tensor boundaries.
# ruff: noqa
# fmt: off
while sample_number > 0:
    all_token = ms_tokenizer.toks
    with torch.no_grad():
        for i in tqdm(nano_loc, total=len(nano_loc), desc='Nanobody Humanization Process'):
            nano_prediction = model(
                nano_pad_token.to(device),
                nano_pad_region.to(device),
                H_chn_type=None
            )

            nano_pred = nano_prediction[:, i, :len(all_token)-1]
            nano_soft = torch.nn.functional.softmax(nano_pred, dim=1)
            nano_sample = torch.multinomial(nano_soft, num_samples=1)
            nano_pad_token[:, i] = nano_sample.squeeze()

    nano_untokenized = [ms_tokenizer.idx2seq(s) for s in nano_pad_token]
    for _, g_h in enumerate(nano_untokenized):
        if sample_number == 0:
            break

        with open(save_fpath, 'a', encoding='UTF-8') as f:
            # try:
            sample_origin = 'humanization'
            sample_name = str(pdb_name)
            # Make sure that the sample seq can be detected by the Chain.
            # Duplicated.
            if g_h not in duplicated_set:
                test_chain = Chain(g_h, scheme='imgt')
                f.write(f'{sample_origin},{sample_name},{g_h}\n')
                duplicated_set.add(g_h)
                sample_number -= 1
                logger.info('Already Sample number {}'.format(args.sample_number - sample_number))
                logger.info('Sample Heavy Chain Seq: {}'.format(g_h))
            else:
                sample_number -= 1

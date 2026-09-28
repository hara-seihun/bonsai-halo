#!/usr/bin/env python3
import unittest

from prefill_budget import budget


class WorkloadContract(unittest.TestCase):
    def panel(self, sequences):
        return budget(32, 512, 2.9, 242, 1, 32, 8, sequences=sequences)

    def test_independent_sequences_do_not_share_state_or_prefix(self):
        prompt, generation = self.panel(1), self.panel(32)
        self.assertEqual(prompt['dense_macs_per_token'], generation['dense_macs_per_token'])
        self.assertEqual(generation['rows_per_sequence'], 1)
        self.assertLess(generation['components_macs_per_token']['attention_qk_av'],
                        prompt['components_macs_per_token']['attention_qk_av'])
        self.assertLess(generation['conditional_bandwidth_tps']['chunk_state_unique_kv_and_logits'],
                        prompt['conditional_bandwidth_tps']['chunk_state_unique_kv_and_logits'])
        bandwidth = generation['conditional_bandwidth_tps']
        self.assertEqual(bandwidth['weights_and_per_token_f32_state'],
                         bandwidth['weights_and_once_per_chunk_f32_state'])

    def test_precision_choices_remain_distinct(self):
        for sequences in [1, 2, 4, 8, 16, 32]:
            c = self.panel(sequences)['conditional_compute_tps']
            self.assertLess(c['all_iu8'], c['ffn_iu4_other_iu8'])
            self.assertLess(c['ffn_iu4_other_iu8'], c['ffn_and_sequence_iu4_head_iu8'])
            self.assertLess(c['ffn_and_sequence_iu4_head_iu8'], c['all_iu4'])

    def test_uneven_or_empty_sequence_groups_are_not_silently_rounded(self):
        for sequences in [0, -1, 3, 33]:
            with self.assertRaises(ValueError):
                self.panel(sequences)


if __name__ == '__main__':
    unittest.main()

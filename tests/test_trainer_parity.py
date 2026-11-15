"""브라우저 학습 엔진(web/trainer.js)을 PyTorch(tinylm)와 대조한다. node 가 없으면 건너뛴다."""
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NODE = shutil.which("node")


def lr_at(step, steps, lr):  # tinylm.train 과 같은 스케줄
    warmup = min(100, max(1, steps // 10))
    if step <= warmup:
        return lr * step / warmup
    frac = (step - warmup) / max(1, steps - warmup)
    min_lr = lr * 0.1
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))


@unittest.skipUnless(NODE, "node 가 없어 건너뜀")
class TrainerParityTest(unittest.TestCase):
    def test_parity_with_pytorch(self):
        # 기본 설정 + 홀수 크기(블로킹 꼬리 경로) 설정
        for arch, B, T in (({"d_model": 32, "n_layer": 2, "n_head": 4, "seq_len": 16, "vocab_size": 50}, 4, 16),
                           ({"d_model": 36, "n_layer": 1, "n_head": 4, "seq_len": 9, "vocab_size": 51}, 3, 7)):
            with self.subTest(arch=arch):
                self._parity(arch, B, T)

    def _parity(self, arch, B, T):
        try:
            import torch
            import torch.nn.functional as F
            from learnbox import tinylm
        except ImportError:
            self.skipTest("torch 없음")
        steps, lr = 100, 3e-3
        torch.manual_seed(0)
        model = tinylm.build_model(arch)
        model.fast = False
        with torch.no_grad():  # bias·LayerNorm 경로도 검증하도록 흔든다
            for n, p in model.named_parameters():
                if n.endswith("bias") or ".ln" in n or n.startswith("ln."):
                    p.add_(0.05 * torch.randn_like(p))
        names = [n for n in model.state_dict() if n != "head.weight"]
        weights = {n: model.state_dict()[n].detach().reshape(-1).tolist() for n in names}
        g = torch.Generator().manual_seed(3)
        data = torch.randint(4, arch["vocab_size"], (B, T + 1), generator=g)
        x, y = data[:, :-1], data[:, 1:]
        model.train()
        loss = F.cross_entropy(model(x).reshape(-1, arch["vocab_size"]), y.reshape(-1))
        loss.backward()
        pmap = dict(model.named_parameters())
        grads = {n: pmap[n].grad.detach().reshape(-1).tolist() for n in names}
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        step_lr = lr_at(1, steps, lr)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=step_lr, betas=(0.9, 0.95))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        updated = {n: pmap[n].detach().reshape(-1).tolist() for n in names}
        fx = {"arch": arch, "weights": weights, "x": x.reshape(-1).tolist(), "y": y.reshape(-1).tolist(),
              "B": B, "T": T, "lr": lr, "step": 1, "steps": steps,
              "expected": {"loss": loss.item(), "grads": grads, "updated": updated}}
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "fixture.json"
            fp.write_text(json.dumps(fx))
            r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "trainer_parity.js"), str(fp)],
                               capture_output=True, text=True, timeout=60)
        self.assertTrue(r.stdout.strip(), r.stderr)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        print("parity:", res)
        self.assertLessEqual(abs(res["loss"] - loss.item()), 1e-3)
        self.assertLessEqual(res["max_rel_grad_err"], 1e-2)
        self.assertLessEqual(res["max_rel_update_err"], 1e-2)
        self.assertTrue(res["ok"])

    def test_js_unit_tests(self):
        r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "trainer_unit.test.js")],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()

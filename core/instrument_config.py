from dataclasses import dataclass

from core.hardware_base import validate_distinct_addresses, validate_terminal


UNSELECTED_TEXT = '未选择'


@dataclass
class InstrumentSettings:
    bias_address: str = ''
    bias_terminal: str = 'REAR'
    gate_address: str = ''
    gate_terminal: str = 'REAR'

    def update(self, bias_address, bias_terminal, gate_address, gate_terminal):
        self.bias_address = self._clean_address(bias_address)
        self.bias_terminal = str(bias_terminal).strip().upper() or 'REAR'
        self.gate_address = self._clean_address(gate_address)
        self.gate_terminal = str(gate_terminal).strip().upper() or 'REAR'

    def snapshot(self, require_gate=False):
        if not self.bias_address:
            raise ValueError('请先在首页选择偏压表')
        validate_terminal(self.bias_terminal, '偏压端子')
        if require_gate:
            if not self.gate_address:
                raise ValueError(
                    '该测试需要偏压表和栅压表，目前仅检测到 1 台'
                )
            validate_terminal(self.gate_terminal, '栅压端子')
            validate_distinct_addresses(
                self.bias_address, self.gate_address, True
            )
        return {
            'bias_address': self.bias_address,
            'bias_terminal': self.bias_terminal,
            'gate_address': self.gate_address,
            'gate_terminal': self.gate_terminal,
        }

    def to_config(self):
        return {
            'bias': {
                'address': self.bias_address,
                'terminal': self.bias_terminal,
            },
            'gate': {
                'address': self.gate_address,
                'terminal': self.gate_terminal,
            },
        }

    def load_config(self, value):
        if not isinstance(value, dict):
            return
        bias = value.get('bias', {})
        gate = value.get('gate', {})
        if not isinstance(bias, dict) or not isinstance(gate, dict):
            return
        self.update(
            bias.get('address', ''),
            bias.get('terminal', 'REAR'),
            gate.get('address', ''),
            gate.get('terminal', 'REAR'),
        )

    @staticmethod
    def _clean_address(value):
        text = str(value or '').strip()
        return '' if text == UNSELECTED_TEXT else text

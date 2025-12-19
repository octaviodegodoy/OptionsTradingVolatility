from constants import CALL_OPTION, PUT_OPTION, TYPE_BUY, TYPE_SELL


def calculate_position_deltas(pos, option_type, delta):
                """
                Calculate delta contribution for a position.
                
                Args:
                    pos: Position object with volume and type
                    option_type: CALL_OPTION or PUT_OPTION
                    delta: Calculated delta value
                
                Returns:
                    tuple: (delta_calls_contribution, delta_puts_contribution)
                """
                delta_calls = 0.0
                delta_puts = 0.0
                
                if option_type == CALL_OPTION:
                    if pos.type == TYPE_BUY:
                        delta_calls = delta * pos.volume
                    elif pos.type == TYPE_SELL:
                        delta_calls = -delta * pos.volume
                
                elif option_type == PUT_OPTION:
                    if pos.type == TYPE_BUY:
                        delta_puts = -delta * pos.volume
                    elif pos.type == TYPE_SELL:
                        delta_puts = delta * pos.volume
                
                return delta_calls, delta_puts
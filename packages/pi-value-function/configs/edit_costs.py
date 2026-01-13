
# example usage:
# set all failure costs to 2000
# uv run edit_costs.py --set_all 2000
# increase all failure costs by 10%
# uv run edit_costs.py --increase_cost_percent 1.1
# increase all failure costs by 10 points
# uv run edit_costs.py --increase_cost 10

from pathlib import Path
import json

script_dir = Path(__file__).parent
failure_cost_json = script_dir / "failure_costs.json"

def load_failure_costs():
    with open(failure_cost_json, "r") as f:
        return json.load(f)
def save_failure_costs(costs):
    with open(failure_cost_json, "w") as f:
        json.dump(costs, f, indent=4)
def set_all_costs(new_cost: float):
    costs = load_failure_costs()
    for item in costs:
        item["c_fail"] = new_cost
    save_failure_costs(costs)
    print(f"Set all failure costs to {new_cost}")
def increase_costs_by_percent(factor: float):
    costs = load_failure_costs()
    for item in costs:
        item["c_fail"] *= factor
    save_failure_costs(costs)
    print(f"Increased all failure costs by a factor of {factor}")
def increase_costs_by_amount(amount: float):
    costs = load_failure_costs()
    for item in costs:
        item["c_fail"] += amount
    save_failure_costs(costs)
    print(f"Increased all failure costs by {amount}")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Edit failure costs in JSON config.")
    parser.add_argument("--set_all", type=float, help="Set all failure costs to this value.")
    parser.add_argument("--increase_cost_percent", type=float, help="Increase all failure costs by this percentage (e.g., 1.1 for +10%).")
    parser.add_argument("--increase_cost", type=float, help="Increase all failure costs by this fixed amount.")

    args = parser.parse_args()

    if args.set_all is not None:
        set_all_costs(args.set_all)
    elif args.increase_cost_percent is not None:
        increase_costs_by_percent(args.increase_cost_percent)
    elif args.increase_cost is not None:
        increase_costs_by_amount(args.increase_cost)
    else:
        print("No action specified. Use --set_all, --increase_cost_percent, or --increase_cost.")
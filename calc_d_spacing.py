#!/usr/bin/env python3
"""
Interactive test script for calc_d_hexagonal function
"""
from Functions_StructureFactor import calc_d_hexagonal

def test_d_spacing():
    """Test the calc_d_hexagonal function interactively"""
    
    # FePSe3 lattice parameters
    a = 6.0155
    c = 15.3415
    
    print("Testing calc_d_hexagonal function")
    print(f"Lattice parameters: a = {a} Å, c = {c} Å")
    print("="*50)
    
    print("\n" + "="*50)
    print("Interactive mode - Enter your own h,k,l values")
    print("Type 'quit' or 'exit' to stop")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter h,k,l (e.g., 1,2,3) or 'quit': ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Parse input
            h, k, l = map(int, user_input.split(','))
            
            # Calculate d-spacing
            d = calc_d_hexagonal(h, k, l, a, c)
            print(f"({h},{k},{l}): d = {d:.4f} Å")
            
            # # Additional info
            # print(f"  1/d² = {1/d**2:.6f} Å⁻²")
            
        except ValueError:
            print("Invalid input! Please enter three integers separated by commas (e.g., 1,0,1)")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_d_spacing()

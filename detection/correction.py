class Corrector:
    def __init__(self):
        pass


    def fix_new_rack_numbers_character(self, rack_number):
        """
        Fixes OCR misinterpretations in a rack number string.

        Args:
            rack_number (str): The misinterpreted rack number.

        Returns:
            str: The corrected rack number.
        """
        if len(rack_number) < 10:  # Ensure we have at least 10 characters to modify
            raise ValueError("Rack number is too short to process!")

        # Convert to list for in-place modifications
        rack_chars = list(rack_number)

        # Dictionary mapping incorrect characters to their corrections
        corrections = {
            2: {'O': '0', 'U': '0'},
            3: {'M': '1'},
            5: {'I': '/', '1': '/'},
            6: {'8': 'B'},
            7: {'I': '/', '1': '/'},
            8: {'O': '0', 'U': '0'},
            9: {'S': '5', 'I': '1', 'O': '0'}
        }

        # Apply corrections
        for index, fix_map in corrections.items():
            if rack_chars[index] in fix_map:
                rack_chars[index] = fix_map[rack_chars[index]]

        # Convert list back to string
        return ''.join(rack_chars)


    def fix_part_number(self, text):
        """
        Corrects OCR misinterpretations in a part number string.

        Args:
            text (str): The misinterpreted part number.

        Returns:
            str: The corrected part number.
        """
        if len(text) < 6:  # Ensure at least 6 characters to prevent indexing errors
            return text

        # Check if the last character is a letter (indicating part number format)
        if text[-1].isalpha():
            number = list(text[-5:-1])  # Extract the last 4-digit part number

            # Mapping of incorrect OCR characters to correct ones
            corrections = {'G': '6', 'O': '0', 'o': '0', 'A': '4', 'Z': '2', 
                        'S': '5', 'I': '1', 'T': '1'}

            # Apply corrections
            number = [corrections.get(char, char) for char in number]

            # Reconstruct the string
            new_text = text[:-5] + ''.join(number) + text[-1]

            # Debugging Output (Remove if not needed)
            print("Original:", text)
            print("Fixed:", new_text)

            return new_text

        return text


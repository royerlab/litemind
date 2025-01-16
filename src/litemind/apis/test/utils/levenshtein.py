def levenshtein_distance(s1: str, s2: str):
    """
    Compute the Levenshtein distance between two strings.

    Parameters
    ----------
    s1: str
        The first string
    s2: str
        The second string

    Returns
    -------
    int
        The Levenshtein distance between the two strings

    """

    # If the strings are of different lengths, ensure s1 is the longer one
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # If one of the strings is empty, return the length of the other string
    if len(s2) == 0:
        return len(s1)

    # Create a list of integers from 0 to the length of s2
    previous_row = range(len(s2) + 1)

    # Iterate over the characters in s1
    for i, c1 in enumerate(s1):

        # Create a list with the first element being the index of the current character in s1
        current_row = [i + 1]

        # Iterate over the characters in s2
        for j, c2 in enumerate(s2):
            # Calculate the cost of insertions, deletions, and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            # Append the minimum cost to the current row
            current_row.append(min(insertions, deletions, substitutions))

        # Set the previous row to the current row
        previous_row = current_row

    # Return the last element in the previous row
    return previous_row[-1]

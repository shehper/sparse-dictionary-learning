import re

# TODO: improve comments in this file.
# Also perhaps, this workflow can be simplified. 

def replace_html_chars(text):
    """This function takes a string and replaces HTML special characters with their corresponding HTML entities but preserves <br> as it is."""
    text = text.replace("<br>", "PLACEHOLDER_FOR_BR") # replace all instances of <br> with a placeholder
    html_escape_table = {"&": "&amp;", '"': "&quot;", "'": "&apos;", ">": "&gt;", "<": "&lt;"} # get a dictionary of HTML special characters and their entities
    escaped_text = "".join(html_escape_table.get(c, c) for c in text) # replace each special character with its corresponding HTML entity
    escaped_text = escaped_text.replace("PLACEHOLDER_FOR_BR", "<br>") #replace the placeholder with <br> again
    return escaped_text

def print_dict(mydict, sort=False, neuron_descriptions=None):
    """This function prints the dictionary of top activations in a way that is suitable to be rendered as HTML text later"""
    items = list(mydict.items())
    if sort: # if sort is True, sort neurons by their highest activation values
        items.sort(key=lambda x: x[1][0], reverse=True)
    
    # initiate a string that will contain all of the text
    output = "" 

    # iterate over all neurons and its list of top activations
    for (neuron_id, list_of_activations_and_contexts) in items:
        
        # write a header for each neuron
        output += f'<b> Neuron: {neuron_id} </b> <br> <br>  ' 

        # if we have a description for this neuron, add that to the text
        if neuron_descriptions and neuron_id in neuron_descriptions: 
            output += f'<span style="color:blue;"> Common Theme: {neuron_descriptions[neuron_id]}  </span> <br> <br>'

        # now get the contribution of current neuron to the string
        current_neuron_str = ""
        # update the current neuron string with contexts and activations for current neuron
        for activation, context in list_of_activations_and_contexts:
            current_neuron_str += f'{context} ; {activation:.4f} <br>'

        # if there is any HTML text in the current neuron string, make sure to replace it so that the HTML output is not disturbed
        for char in ["&", '"', "'", ">", "<"]:
            if char in current_neuron_str:
                current_neuron_str = replace_html_chars(current_neuron_str)
                break
        output += current_neuron_str + '<br> <br> <hr> <br>'
    print(output)

def ansi_to_html(text):
    """This function replaces the ANSI red color wrappers in python strings, i.e. \x1b\[91m and \x1b\[0m with corresponding HTML """
    text = re.sub(r'\x1b\[91m', '<span style="color:red;">', text) # convert ANSI color codes to HTML
    text = re.sub(r'\x1b\[0m', '</span>', text)
    text = text.replace('\n', '&#x23CE;') # replace newline characters with HTML line breaks
    return text

if __name__ == '__main__':
    raise NotImplementedError # TODO: complete this code

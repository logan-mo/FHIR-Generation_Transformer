import json
import re

def string_to_dict(s):
    split_list = re.findall('\[.*?\]\s[^\[]*|\[.*?\]\s|\s', s)
    result = {}

    for item in split_list:
        key, value = item.split('] ')
        keys = key.replace("][", "|").strip("[").split("|")
        temp = result
        for k in keys[:-1]:
            if not k.isdigit():
                if k not in temp:
                    # If the next key is a number, then this key should be a list
                    if keys[keys.index(k) + 1].isdigit():
                        temp[k] = []
                    else:
                        temp[k] = {}
                temp = temp[k]
            else:
                if len(temp) <= int(k):
                    temp.append({})  # append a new dictionary to the list
                temp = temp[int(k)]  # Go to the dictionary at index k in the list

        if isinstance(temp, list):
            if len(temp) <= int(keys[-1]):
                temp.append(value.strip())  # append the new value and strip whitespace
            else:  
                temp[int(keys[-1])] = value.strip()  # if the index already exists, replace the value and strip whitespace
        else:
            temp[keys[-1]] = value.strip()  # strip whitespace

    return result



def dict_to_json(dict_obj):
    return json.dumps(dict_obj)


s = "[resourceType] AllergyIntolerance [id] 0a8f3194-8246-4c16-a93d-ee257c5290dd [clinicalStatus][coding][0][system] http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical [clinicalStatus][coding][0][code] active [verificationStatus][coding][0][system] http://terminology.hl7.org/CodeSystem/allergyintolerance-verification [verificationStatus][coding][0][code] confirmed [type] allergy [category][0] food [criticality] high [code][coding][0][system] http://snomed.info/sct [code][coding][0][code] 91935009 [code][coding][0][display] Allergy to peanuts (disorder) [patient][reference] Patient/9766d024-dbe6-7572-64ae-965f69298c49 [onsetDateTime] 2000-01-01T00:00:00-04:00 [recordedDate] 2023-07-02T00:00:00-04:00 [recorder][reference] Practitioner/1234 [asserter][reference] Patient/9766d024-dbe6-7572-64ae-965f69298c49 [reaction][0][substance][coding][0][system] http://snomed.info/sct [reaction][0][substance][coding][0][code] 91935009 [reaction][0][substance][coding][0][display] Allergy to peanuts (disorder)"
fhir_dict = string_to_dict(s)
json_string = dict_to_json(fhir_dict)
print(json_string)
#print(fhir_dict)

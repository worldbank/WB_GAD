import sys, os
import requests
import json

import geopandas as gpd
import pandas as pd

from shapely.geometry import Point, Polygon, box, shape

def get_wb_classifications(grouping_version='38.0', fmr_prod_base_url = "https://fmr.worldbank.org/FMR/sdmx/v2/structure/"): 
    """ Extract official World Bank regions and income classifications from Data360

    Parameters
    ----------
    grouping_version : str, optional
        Version of the Data360 FMR to investigate, default is 38.0, which is for FY26
    fmr_prod_base_url : str, optional
        url from which to access Data360 FMR, default is "https://fmr.worldbank.org/FMR/sdmx/v2/structure/"

    Returns
    ----------
    pandas.DataFrame
        Dataframe containing the group, value, and ISO3. This is a LONG format table
    """
    ref_area_groups_url = f"{fmr_prod_base_url}hierarchy/WB/H_REF_AREA_GROUPS/{grouping_version}?format=fusion-json"
    # Extracting hierarchical codes from hierarchy
    response = requests.get(ref_area_groups_url).json()
    
    code_data = []
    for item in response["Hierarchy"][0]["codes"]:
        for sub_item in item["codes"]:
            if "codes" in sub_item:
                for sub2_item in sub_item["codes"]:
                    code_data.append({"Code_URN": sub2_item["urn"].split(")")[-1]})
    
    res = pd.DataFrame(code_data)
    res_split = (
        res["Code_URN"].str.split(".", expand=True).drop(columns=[0])
    )
    res_split = res_split.rename(
        columns={1: "group", 2: "value", 3: "ISO3"}
    )
    return(res_split)

def get_wb_classifications_strict(
    grouping_version="38.0",
    region_version="2.0",
    income_version="2.0",
    fmr_prod_base_url="https://fmr.worldbank.org/FMR/sdmx/v2/structure/",
    type_to_keep=("CONTINENT", "REGION", "INCOME"),
):
    """
    Fetch World Bank classifications (continent, region, income) mapped to ISO3 codes.

    Parameters
    ----------
    grouping_version : str, optional
        Version of the World Bank reference area groups hierarchy. Default is "38.0"
        (corresponding to FY26).
    region_version : str, optional
        Version of the World Bank regions hierarchy. Default is "2.0".
    income_version : str, optional
        Version of the World Bank income groups hierarchy. Default is "2.0".
    fmr_prod_base_url : str, optional
        Base URL for accessing the World Bank FMR SDMX structures. Default is
        "https://fmr.worldbank.org/FMR/sdmx/v2/structure/".
    type_to_keep : tuple of str, optional
        Classification types to keep in the final table. Default is
        ("CONTINENT", "REGION", "INCOME").

    Returns
    -------
    pandas.DataFrame
        A wide-format dataframe containing ISO3 country codes as rows and selected
        classification types (e.g., CONTINENT, REGION, INCOME) as columns.
    """

    ref_area_groups_url = (
        f"{fmr_prod_base_url}hierarchy/WB/H_REF_AREA_GROUPS/{grouping_version}?format=fusion-json"
    )
    region_hierarchy_url = (
        f"{fmr_prod_base_url}hierarchy/WB/H_WB_REGIONS/{region_version}?format=fusion-json"
    )
    income_hierarchy_url = (
        f"{fmr_prod_base_url}hierarchy/WB/H_WB_INCOME/{income_version}?format=fusion-json"
    )

    # Parse H_REF_AREA_GROUPS hierarchy
    response = requests.get(ref_area_groups_url).json()
    code_data = [
        {"Code_URN": sub2["urn"].split(")")[-1]}
        for item in response["Hierarchy"][0]["codes"]
        for sub in item.get("codes", [])
        for sub2 in sub.get("codes", [])
    ]

    # Split URNs into structured columns
    res_split = (
        pd.DataFrame(code_data)["Code_URN"]
        .str.split(".", expand=True)
        .rename(columns={1: "group", 2: "value", 3: "ISO3"})
        .drop(columns=[0])
    )

    # Get valid region codes 
    region_response = requests.get(region_hierarchy_url).json()
    region_codes = [
        sub["id"]
        for item in region_response["Hierarchy"][0]["codes"]
        for sub in item.get("codes", [])
    ]

    # Get valid income codes 
    income_response = requests.get(income_hierarchy_url).json()
    income_codes = [item["id"] for item in income_response["Hierarchy"][0]["codes"]]

    # Filter invalid REGION / INCOME values 
    filters = {
        "REGION": set(region_codes),
        "INCOME": set(income_codes),
    }
    for grp, valid_values in filters.items():
        res_split = res_split[
            ~((res_split["group"] == grp) & (~res_split["value"].isin(valid_values)))
        ]

    # Pivot to wide format
    wide = (
        res_split.pivot_table(
            index="ISO3",
            columns="group",
            values="value",
            aggfunc=lambda x: ",".join(sorted(set(x))),
        )
        .reset_index()
    )

    # Keep only requested groups
    return wide[["ISO3"] + [col for col in type_to_keep if col in wide.columns]]

def merge_id_columns(in_df, col_defs, drop_orig=False):
    """ Combine the two columns in col_defs 

    Parameters
    ----------
    in_df : pandas.DataFrame
        data frame of admin boundaries in which to generate primary key columns
    col_defs : list
        List of the column sets required to create official admin primary key
        [['P_CODE_1', 'P_CODE_1_t'], ['ADM1CD', 'ADM1CD_t]']
    drop_orig : bool, optional
        Whether to drop the original columns after merging, by default False

    Returns
    -------
    pandas.DataFrame
        in_df with columns in col_defs combined to create a primary key
    """
    for col_def in col_defs:
        out_col = col_def[-1][:-1] + "c"
        in_df[out_col] = in_df[col_def[0]]
        in_df.fillna({out_col: in_df[col_def[1]]}, inplace=True)
        if drop_orig:
            in_df.drop(columns=col_def, inplace=True)
    return in_df

def check_duplicates(in_df, id_col, out_file):
    """ Look for duplicates in in_df, based on the id_col

    Parameters
    ----------
    in_df : pandas.DataFrame
        data frame of admin boundaries in which to generate primary key columns
    id_col : str
        column in in_df containing primary key
    out_file : str (file path)
        where to write broken data
    """
    
    adm1_dups = in_df[id_col].duplicated().sum()
    print(f"{id_col} duplicates: {adm1_dups}")
    if adm1_dups > 0:
        # write duplicated records in adm1 and adm2 to QA/QC folder
        adm1_dups = in_df[merged_adm1.duplicated(subset=[id_col], keep=False)]
        adm1_dups.to_file(out_file, driver='GPKG')

def evaluate_duplicate_names(in_df, name_col, parent_col, log_file):
    """ Group features by a parent_col and check if any names are duplicated

    Parameters
    ----------
    in_df : _geopandas.GeoDataFrame_
        administrative bounds of interest
    name_col : str
        column containing names to check for duplicates
    parent_col : str
        column containing parent administrative unit code to group by
    log_file : str
        path to the log file where duplicates will be recorded
    """
    original_stdout = sys.stdout
    with open(log_file, 'w') as log_fh:
        log_fh.write(f"Checking for duplicate names in {name_col} grouped by {parent_col}\n")
        log_fh.write("--------------------------------------------------\n")
        sys.stdout = log_fh
        for label, group in in_df.groupby(parent_col):
            if group[name_col].duplicated().any():
                print(f"Duplicates found in {label} for {name_col}:")                
                print(group[group[name_col].duplicated(keep=False)][[name_col, parent_col]])                
            else:
                #print(f"No duplicates found in {label} for {name_col}.")
                pass                
    sys.stdout = original_stdout


def open_and_write_to_better_formats(filename, out_folder):
    """ RETIRED Open a GeoDataFrame and write it to a better format (GPKG) if it doesn't already exist.

    Parameters
    ----------
    filename : str
        path to existing 
    out_folder : str
        url from which to access Data360 FMR, default is "https://fmr.worldbank.org/FMR/sdmx/v2/structure/"

    Returns
    ----------
    pandas.DataFrame
        Dataframe containing the group, value, and ISO3. This is a LONG format table       
    """
    out_file = os.path.join(out_folder, os.path.basename(filename).replace(".shp", ".gpkg"))
    if not os.path.exists(out_file):
        gdf = gpd.read_file(filename)
        gdf.to_file(out_file, driver='GPKG')
        print(f'Wrote {out_file}')
    else:
        gdf = gpd.read_file(out_file)
        print(f'{out_file} already exists, skipping write.')
    return gdf


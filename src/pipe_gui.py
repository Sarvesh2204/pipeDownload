import ipywidgets as widgets
import pipe_inventory
import pandas as pd
pd.set_option("display.max_rows", None)
from IPython.display import HTML, display

def show_main_gui(df_inventory):
    program_type_widget = widgets.Text(
        description="ProgramType",
        placeholder="e.g. 10"
    )

    program_id_widget = widgets.Text(
        description="ProgramID",
        placeholder="e.g. 2"
    )

    obsid_widget = widgets.Text(
        description="ObsID",
        placeholder="e.g. 4"
    )

    visit_counter_widget = widgets.Text(
        description="VisitCounter",
        placeholder="e.g. 1"
    )
    
    target_widget = widgets.Text(
    description="Target",
    placeholder="e.g. WASP")

    visitid_widget = widgets.Text(
        description="VisitID",
        placeholder="e.g. 100041000401"
    )


    search_button = widgets.Button(
        description="Search",
        button_style="success",
        icon="search"
    )

    reset_button = widgets.Button(
        description="Reset",
        button_style="warning",
        icon="refresh"
    )
    
    download_button = widgets.Button(
    description="Download Selected",
    button_style="primary",
    icon="download"
    )

    output = widgets.Output()
    download_output = widgets.Output()



    visit_selector = widgets.SelectMultiple(
        options=[],
        description="Visits",
        rows=15,
        layout=widgets.Layout(
            width="1200px",
            height="350px"
            )    )
    
    table_output = widgets.Output(
    layout=widgets.Layout(
        height="400px",
        overflow="auto",
        border="1px solid lightgray"
    )
)



    def run_search(_):
        pd.set_option("display.max_rows", None)

        result = pipe_inventory.filter_inventory(
            df_inventory,
            program_type=program_type_widget.value.strip() or None,
            program_id=program_id_widget.value.strip() or None,
            obsid=obsid_widget.value.strip() or None,
            visit_counter=visit_counter_widget.value.strip() or None,
            target=target_widget.value.strip() or None,
            visit_id=visitid_widget.value.strip() or None,
        )

        result = result.sort_values(
            [
                "ProgramType",
                "ProgramID",
                "ObsID",
                "VisitCounter"
            ]
        ).reset_index(drop=True)
        visit_selector.options = (
            pipe_inventory.build_visit_options(result)
        )
        with output:

            output.clear_output()

            print(f"{len(result)} visits found")

        with table_output:

            table_output.clear_output()

            display(result)

            
            
    def reset_search(_):

        # ---------------------
        # Clear search fields
        # ---------------------

        program_type_widget.value = ""
        program_id_widget.value = ""
        obsid_widget.value = ""
        visit_counter_widget.value = ""
        visitid_widget.value = ""

        # ---------------------
        # Clear selector
        # ---------------------

        # visit_selector.options = []
        visit_selector.options = (
        pipe_inventory.build_visit_options(
        df_inventory))

        # ---------------------
        # Restore visit count
        # ---------------------

        with output:

            output.clear_output()

            print(
                f"{len(df_inventory)} visits found"
            )

        # ---------------------
        # Restore full table
        # ---------------------

        with table_output:

            table_output.clear_output()

            display(df_inventory)

    def download_selected(_):
        from IPython.display import HTML, display


        with download_output:

            download_output.clear_output()

            selected_visits = list(
                visit_selector.value
            )

            # --------------------------------
            # Nothing selected
            # --------------------------------

            if len(selected_visits) == 0:

                print(
                    "Please select at least one visit."
                )

                return

            # --------------------------------
            # Status message
            # --------------------------------

            print(
                f"Preparing ZIP for "
                f"{len(selected_visits)} visit(s)..."
            )

            print(
                "Please wait..."
            )

            # Force immediate display update
            display(
                widgets.HTML(
                    "<b>Creating ZIP archive...</b>"
                )
            )

            # --------------------------------
            # Create ZIP
            # --------------------------------

            zip_path = pipe_inventory.create_visit_zip(
                df_inventory,
                selected_visits
            )

            # --------------------------------
            # Show download link
            # --------------------------------

            from IPython.display import (
                HTML,
                display
            )

            download_output.clear_output()

            print(
                f"{len(selected_visits)} visit(s) added to ZIP."
            )

            print(
                f"ZIP file: {zip_path}"
            )

            print(
                "Click the link below to download."
            )

            display(
                HTML(
                    f"""
                    <a href="../../{zip_path}" download>
                        Download ZIP
                    </a>
                    """
                )
            )

    # Events        
    search_button.on_click(run_search)
    reset_button.on_click(reset_search)
    download_button.on_click(download_selected)



    display(
        program_type_widget,
        program_id_widget,
        obsid_widget,
        visit_counter_widget,
        target_widget,
        visitid_widget,
        widgets.HBox([search_button, reset_button,download_button]),
        output,
        widgets.HTML("<h4>Select Visits</h4>"),
        visit_selector,
        download_output,
        widgets.HTML("<h4>PIPE Results</h4>"),
        table_output,
        
    )


    with output:

        output.clear_output()

        print(f"{len(df_inventory)} visits found")
        
    with table_output:

        table_output.clear_output()

        display(df_inventory)
        
    visit_selector.options = (
    pipe_inventory.build_visit_options(
        df_inventory
    ))
    

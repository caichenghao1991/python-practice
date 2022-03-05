'''
    Power BI desktop (create)
        authoring tool, create power bi models, reports

        Home -> Get Data (excel, sql server, mysql, oracle, csv, json, power bi dataset/dataflow...)
            -> Navigator (select tables and load,
                select tables for edit (open data in a power query editor) if necessary,
                this create a copy of data)

            inside right side fields panel (filter, visualizations, fields) will see table name, fields
            inside left side data panel (report, data, relation view) will see table data

            need clean up human friendly but not machine friendly dataset (extra rows/columns for total(calculation))
            click under home tab to refresh to apply query editor applied steps to new dataset

            save as .pbix file  very compressed can handle 100 millions data
            can create many report based on one model
            Publish model to cloud

        Modeling tab has sort column by date, and formatting to change decimal places and currency

    Query Editor
        on the right side applied steps remove changed type and promoted header since most likely wrong header
        At top home tab has remove rows/columns, use first row as headers, close and apply(save and exit query editor)
        Applied steps will track changes done and have each state recorded. allow change each step later on with gear
            icon, delete step with cross icon and add additional step between old steps (do as normal modification and
            will ask whether insert step)
        column name arrow button has many options (filters(equal, begin with, end with, contain, and not previous),sort)
        right click column has copy, remove, duplicate column, remove duplicate, change type, transform, replace values,
        group by, unpivot, rename, move... options
        right side column name can click and change data type


    Data Table: record transaction (many rows)
    Lookup Table: define who, what, when, where, how
    left side relation view panel add relation by click one key drag into other table same (foreign) key (direction not
        matter). result has 1 at one side table edge, leading to an arrow towards many side table

    left side report panel add visualization charts with right side visualization panel (different charts, and fields
        (expand right side fields panel table name, drag relevant field into visualization fields, drop down option of
        dragged fields has: don't summarize, other aggregation functions) / format (font, style
        grid, values, general,..., also can just search in the text box) /analytics)
        right side filter panel can filter out one page or all pages, drag fields in the fields panel to apply filter
        to that column
        can only apply to chart with columns form one table or linked table, need common column for group by aggregation
        click on column and drill down allow chart on that column field exploded with each value of that column

    data analysis expression
        instead of drag drop fields and add aggregation with visualization field drop down,
        explicit create new measure in top modeling tab, define measure: newColumnName = SUM(TableName[ColumnName])
        hit enter and this will add field in right field panel for further usage
        measure will not change the original dataset, they are calculated dynamically, won't increase file size
        hybrid measure can apply to all tables
        advantage: control, reuse, connected report

        modeling tab new column to add new column with formula: totalSales = DIVIDE(TableName[ColumnName],2)
        hit enter to add new column




























    Power BI website (publish)

'''
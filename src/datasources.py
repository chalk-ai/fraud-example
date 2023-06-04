from chalk.sql import SnowflakeSource, PostgreSQLSource


# These data sources can be used as singletons for querying
# data. The credentials for these data sources have been
# added in the Chalk dashboard under the "Data Sources" tab.
#
# They can alternatively be configured with environment
# variables, or if you have many data sources of the same type,
# you can assign configuration variables to a data source
# name, and provide that name in the constructors below.
snowflake = SnowflakeSource()
postgres = PostgreSQLSource()
